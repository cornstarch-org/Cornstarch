"""Training schedules for pipeline-parallel distributed execution.

A schedule exists **only when pipeline parallelism is used**. Without pipeline
parallelism the modules are co-located on every rank and the training loop runs
the execution plan directly (``output_future.execute()`` + ``backward()``),
accumulating gradients over the user's ``collate_fn`` microbatch list — no
schedule is involved.

When pipeline parallelism is used the modules are disaggregated onto disjoint
rank ranges and form a pipeline:

- each **modality encoder is a leading pipeline stage** (one stage; intra-encoder
  pipelining is out of scope), feeding
- the **language-model stages** (``merge`` runs on the first language-model
  stage; the language model is pipelined across the rest).

The boundary between the encoder mesh and the language-model mesh is a
pipeline-stage boundary — a *seam* — crossed with the same point-to-point
transport as an ordinary stage hop, but with the cross-mesh rank pairing
(producer representative broadcasts to the consumer's first-stage TP/EP group;
mirrored on the backward).

``TrainingSchedule``
    Abstract base: ``step(microbatches, criterion, optimizer)`` runs one
    forward-backward training step over a list of microbatches.

``BasePipelineSchedule``
    Builds, for this rank, its place in the global pipeline (which stage it owns,
    how it forwards, and how it communicates with its neighbors), and the shared
    forward/backward microbatch machinery. Subclasses implement the ordering.

``OneForwardOneBackwardSchedule``
    The 1F1B ordering (warmup / steady / cooldown).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from cornstarch.distributed.pipeline_parallel.p2p import (
    PipelineP2PCommunication,
    exchange_objects,
)
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.models.multimodal.execution import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    _first_output_tensor,
)


# ---------------------------------------------------------------------------
# Tree utilities for nested tensor structures
# ---------------------------------------------------------------------------

def _tree_map(fn: Callable, obj: Any) -> Any:
    """Apply ``fn`` to every tensor in a nested dict / list / tuple."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _tree_map(fn, v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_map(fn, v) for v in obj)
    return obj


def _detach(obj: Any) -> Any:
    """Detach all tensors from the computation graph."""
    return _tree_map(lambda t: t.detach(), obj)


def _first_tensor_device(obj: Any) -> torch.device | None:
    """Return the device of the first tensor found in a nested structure."""
    if isinstance(obj, torch.Tensor):
        return obj.device
    if isinstance(obj, dict):
        for value in obj.values():
            device = _first_tensor_device(value)
            if device is not None:
                return device
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            device = _first_tensor_device(value)
            if device is not None:
                return device
    return None


def _retain_grad(obj: Any) -> None:
    """Call ``retain_grad()`` on tensors that require gradients."""
    def _rg(t: torch.Tensor) -> torch.Tensor:
        if t.requires_grad:
            t.retain_grad()
        return t
    _tree_map(_rg, obj)


def _merge_batch(batches: list[Any], dim: int = 0) -> Any:
    """Concatenate a list of batches along ``dim``."""
    if not batches:
        return None
    if isinstance(batches[0], torch.Tensor):
        return torch.cat(batches, dim=dim)
    if isinstance(batches[0], dict):
        return {k: _merge_batch([b[k] for b in batches], dim) for k in batches[0]}
    return batches


def _default_device() -> torch.device:
    return (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else torch.device("cpu")
    )


# ---------------------------------------------------------------------------
# TrainingSchedule base
# ---------------------------------------------------------------------------

class TrainingSchedule(ABC):
    """Execute one forward-backward training step and return results.

    ``step`` takes the list of microbatches for one optimizer step (the value the
    user's ``collate_fn`` returns), runs forward + loss + backward across the
    pipeline, and returns a dict with ``"loss"`` (accumulated on the last stage,
    ``None`` elsewhere) and ``"outputs"``. The caller owns gradient
    synchronization and the optimizer step.
    """

    @abstractmethod
    def step(
        self,
        microbatches: list[dict[str, torch.Tensor]],
        criterion: Callable[..., Any],
        optimizer: Optimizer | None = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        ...


# ---------------------------------------------------------------------------
# Mesh layout — a distributed-agnostic description of one modality's ranks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeshLayout:
    """The rank assignment of one modality's ``ModalProcessGroupMesh``.

    A ``ModalProcessGroupMesh`` is only *constructed* on the ranks that belong to
    its modality, so a rank cannot ask another modality's mesh which global ranks
    it owns. The seam between an encoder mesh and the language-model mesh needs
    exactly that — to pair producer and consumer ranks it must know, on *every*
    rank, where each modality lives. ``MeshLayout`` is that purely arithmetic
    description (no process groups, no collectives), computed for every module on
    every rank from the same ``(dp, pp, cp, tp, ep)`` reshape the mesh uses.
    """

    global_ranks: tuple[int, ...]
    dp_size: int
    num_pp_stages: int
    cp_size: int
    tp_size: int
    ep_size: int

    def _flat_index(self, dp: int, pp: int, cp: int, tp: int, ep: int) -> int:
        return (
            (((dp * self.num_pp_stages + pp) * self.cp_size + cp) * self.tp_size + tp)
            * self.ep_size
            + ep
        )

    def rank_at(self, dp: int, pp: int, cp: int, tp: int, ep: int) -> int:
        """Return the global rank at a single ``(dp, pp, cp, tp, ep)`` coordinate."""
        return self.global_ranks[self._flat_index(dp, pp, cp, tp, ep)]

    def stage_ranks(self, dp: int, pp: int) -> list[int]:
        """Return every global rank in one ``(dp, pp)`` slice (all cp/tp/ep)."""
        return [
            self.rank_at(dp, pp, cp, tp, ep)
            for cp in range(self.cp_size)
            for tp in range(self.tp_size)
            for ep in range(self.ep_size)
        ]

    @property
    def last_stage(self) -> int:
        return self.num_pp_stages - 1

    @property
    def rankset(self) -> frozenset[int]:
        return frozenset(self.global_ranks)


# ---------------------------------------------------------------------------
# Base pipeline schedule
# ---------------------------------------------------------------------------

class BasePipelineSchedule(TrainingSchedule):
    """Drive an execution plan across a pipeline that may span several meshes.

    The global pipeline is ``[encoder stages..., language-model stages...]``. This
    rank owns exactly one global stage (the meshes are disjoint): an encoder rank
    owns its encoder's leading stage; a language-model rank owns its language-model
    pipeline stage. The base class resolves that placement, runs this rank's stage
    forward/backward per microbatch, and moves activations to its neighbors —
    using ordinary pipeline P2P for an intra-mesh hop and the cross-mesh *seam*
    transport for the encoder→language-model boundary. Subclasses choose the
    microbatch ordering (1F1B, etc.).

    The plan stays parallelism-agnostic; all rank/stage/transport knowledge lives
    here. The transfer is mathematically transparent — it is a graph break (like
    any pipeline boundary), with the gradient shipped back explicitly.
    """

    def __init__(
        self,
        plan: CornstarchExecutionPlan,
        output_future: ExecutionFuture,
        layouts: dict[int, MeshLayout],
        meshes: dict[int, ModalProcessGroupMesh],
        dp_size: int,
    ) -> None:
        self._plan = plan
        self._output_future = output_future
        self._layouts = layouts
        self._dp_size = dp_size
        self._my_rank = dist.get_rank()
        self._device: torch.device | None = None

        nodes = plan._topological_nodes(output_future.name)
        self._encoder_nodes = [n for n in nodes if n.kind == "run_modality_encoder"]
        self._merge_node = next(
            (n for n in nodes if n.kind == "merge_modality_encoder_outputs"), None
        )
        self._lm_node = next(n for n in nodes if n.kind == "run_language_model")

        self._lm_layout = layouts[id(self._lm_node.params["module"])]
        self._num_encoders = len(self._encoder_nodes)
        self._num_stages = self._num_encoders + self._lm_layout.num_pp_stages

        # Resolve this rank's role and global stage index.
        self._role: str | None = None
        self._encoder_index = -1
        for index, node in enumerate(self._encoder_nodes):
            if self._my_rank in layouts[id(node.params["module"])].rankset:
                self._role = "encoder"
                self._encoder_index = index
                self._encoder_node = node
                self._encoder_layout = layouts[id(node.params["module"])]
                self._stage = index
                break
        if self._role is None and self._my_rank in self._lm_layout.rankset:
            self._role = "llm"
            self._lm_mesh = meshes[id(self._lm_node.params["module"])]
            self._lm_comm = PipelineP2PCommunication(self._lm_mesh)
            self._lm_stage = self._lm_mesh.stage
            self._stage = self._num_encoders + self._lm_stage

        # A rank with no stage in *this* batch's plan idles (e.g. the encoder
        # ranks on a text-only step, whose plan has no run_modality_encoder node).
        # Every rank derives the same plan from the same batch, so the active
        # ranks never wait on a transfer from an idle one.
        self._idle = self._role is None

        # Cross-mesh seam between the (single) leading encoder and LM stage 0.
        # The seam pairs the producing encoder with the language model; multiple
        # encoders feeding one merge as parallel leading stages is out of scope.
        self._has_encoder = self._num_encoders > 0
        if self._has_encoder:
            self._seam_producer = layouts[
                id(self._encoder_nodes[0].params["module"])
            ]
            # The merge consumes the encoder output under this future name.
            self._encoder_output_name = self._encoder_nodes[0].name

        self._modality_token_ids: dict[str, int] = (
            dict(self._merge_node.params.get("modality_token_ids", {}))
            if self._merge_node is not None
            else {}
        )

    # ------------------------------------------------------------------
    # Stage semantics
    # ------------------------------------------------------------------

    @property
    def num_stages(self) -> int:
        return self._num_stages

    @property
    def stage(self) -> int:
        return self._stage

    def is_first_stage(self) -> bool:
        return self._stage == 0

    def is_last_stage(self) -> bool:
        return self._stage == self._num_stages - 1

    def _at_llm_first_stage(self) -> bool:
        return self._role == "llm" and self._lm_stage == 0

    def _seam_on_recv(self) -> bool:
        """This rank receives its forward input across the seam (LM stage 0)."""
        return self._at_llm_first_stage() and self._has_encoder

    def _seam_on_send(self) -> bool:
        """This rank sends its forward output across the seam (the encoder)."""
        return self._role == "encoder"

    # ------------------------------------------------------------------
    # Forward computation for this rank's stage
    # ------------------------------------------------------------------

    def _mask_modality_labels(self, microbatch: dict) -> dict:
        """Return ``microbatch`` with modality-token label positions set to -100.

        Under pipeline parallelism the last language-model stage computes the loss
        from the microbatch labels rather than from ``merge``'s output, so it
        recomputes the same mask locally (every rank has the microbatch and the
        plan). A no-op when there are no modality tokens (text-only / LM-only).
        """
        if not self._modality_token_ids or "input_ids" not in microbatch:
            return microbatch
        input_ids = microbatch["input_ids"]
        labels = microbatch.get("labels")
        if labels is None:
            return microbatch
        token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self._modality_token_ids.values():
            token_mask |= input_ids == int(token_id)
        masked = dict(microbatch)
        masked["labels"] = labels.masked_fill(token_mask, -100)
        return masked

    def _forward_compute(self, microbatch: dict, input_obj: Any | None) -> Any:
        """Run this rank's stage forward and return its output activation/result."""
        if self._role == "encoder":
            output = CornstarchExecutionPlan._execute_node(
                self._encoder_node, dict(microbatch)
            )
            return _first_output_tensor(output)

        # Language-model role.
        if self._lm_stage == 0:
            values = self._mask_modality_labels(microbatch)
            values = dict(values)
            if self._has_encoder:
                # ``input_obj`` is the feature tensor received across the seam.
                values[self._encoder_output_name] = input_obj
            merged = CornstarchExecutionPlan._execute_node(self._merge_node, values)
            values[self._merge_node.name] = merged
            return CornstarchExecutionPlan._execute_node(self._lm_node, values)

        # Later language-model stage: merge the received activation (a dict of
        # pipelined tensors, or a bare hidden_states tensor) into the kwargs and
        # run this stage. The forward spec consumes hidden_states and ignores
        # input_ids off the first stage; labels are used only on the last stage.
        kwargs = dict(self._mask_modality_labels(microbatch))
        if isinstance(input_obj, dict):
            for key in input_obj:
                kwargs.pop(key, None)
            kwargs.update(input_obj)
        elif input_obj is not None:
            kwargs["hidden_states"] = input_obj
        return self._lm_node.params["module"](**kwargs)

    # ------------------------------------------------------------------
    # Forward / backward micro-steps
    # ------------------------------------------------------------------

    def _forward_step(
        self,
        microbatch: dict,
        input_obj: Any | None,
        criterion: Callable,
        num_microbatches: int,
        accum_loss: torch.Tensor | None,
        outputs: list[Any] | None,
    ) -> Any:
        output_obj = self._forward_compute(microbatch, input_obj)
        if self.is_last_stage():
            loss = criterion(output_obj, microbatch) / num_microbatches
            if accum_loss is not None:
                accum_loss.add_(loss.detach())
            if outputs is not None:
                outputs.append(_detach(output_obj))
            return loss
        return output_obj

    def _backward_step(
        self,
        input_obj: Any | None,
        output_obj: Any,
        output_obj_grad: Any | None,
    ) -> Any | None:
        _retain_grad(input_obj)

        if output_obj_grad is None:
            # Last stage: ``output_obj`` is the scalar loss.
            output_obj.backward()
        else:
            keys = None
            if isinstance(output_obj, dict):
                keys = output_obj.get("backward_tensor_keys") or list(
                    output_obj_grad.keys()
                )
            if keys is not None:
                tensors = [
                    output_obj[k] for k in keys if isinstance(output_obj[k], torch.Tensor)
                ]
                grads = [
                    output_obj_grad[k] for k in keys if isinstance(output_obj[k], torch.Tensor)
                ]
                if len(tensors) == 1:
                    torch.autograd.backward(tensors[0], grads[0])
                else:
                    torch.autograd.backward(tensors, grads)
            else:
                torch.autograd.backward(output_obj, output_obj_grad)

        if input_obj is None:
            return None
        if isinstance(input_obj, torch.Tensor):
            return input_obj.grad
        if isinstance(input_obj, dict):
            return {
                k: v.grad
                for k, v in input_obj.items()
                if isinstance(v, torch.Tensor) and v.grad is not None
            }
        return None

    # ------------------------------------------------------------------
    # Seam transport (encoder <-> language-model boundary)
    # ------------------------------------------------------------------

    def _seam_send_forward(self, obj: Any) -> None:
        prod, cons = self._seam_producer, self._lm_layout
        for d in range(self._dp_size):
            producer_rep = prod.rank_at(d, prod.last_stage, 0, 0, 0)
            if self._my_rank == producer_rep:
                exchange_objects(obj, cons.stage_ranks(d, 0), [], self._device)

    def _seam_recv_forward(self) -> Any | None:
        prod, cons = self._seam_producer, self._lm_layout
        for d in range(self._dp_size):
            producer_rep = prod.rank_at(d, prod.last_stage, 0, 0, 0)
            if self._my_rank in cons.stage_ranks(d, 0):
                received = exchange_objects(None, [], [producer_rep], self._device)[0]
                if isinstance(received, torch.Tensor):
                    received.requires_grad_(True)
                return received
        return None

    def _seam_send_backward(self, grad: Any) -> None:
        prod, cons = self._seam_producer, self._lm_layout
        for d in range(self._dp_size):
            consumer_rep = cons.rank_at(d, 0, 0, 0, 0)
            if self._my_rank == consumer_rep:
                exchange_objects(grad, prod.stage_ranks(d, prod.last_stage), [], self._device)

    def _seam_recv_backward(self) -> Any | None:
        prod, cons = self._seam_producer, self._lm_layout
        for d in range(self._dp_size):
            consumer_rep = cons.rank_at(d, 0, 0, 0, 0)
            if self._my_rank in prod.stage_ranks(d, prod.last_stage):
                return exchange_objects(None, [], [consumer_rep], self._device)[0]
        return None

    def _seam_send_forward_recv_backward(self, obj: Any) -> Any | None:
        """Encoder side: send features forward and receive the gradient back.

        Issued as a single ``batch_isend_irecv`` (send + recv together) so the
        encoder and language-model meshes do not both block on a send.
        """
        prod, cons = self._seam_producer, self._lm_layout
        grad = None
        for d in range(self._dp_size):
            producer_rep = prod.rank_at(d, prod.last_stage, 0, 0, 0)
            consumer_rep = cons.rank_at(d, 0, 0, 0, 0)
            if self._my_rank == producer_rep:
                grad = exchange_objects(
                    obj, cons.stage_ranks(d, 0), [consumer_rep], self._device
                )[0]
            elif self._my_rank in prod.stage_ranks(d, prod.last_stage):
                grad = exchange_objects(None, [], [consumer_rep], self._device)[0]
        return grad

    def _seam_send_backward_recv_forward(self, grad: Any) -> Any | None:
        """Language-model side: send the gradient back and receive features.

        Mirror of :meth:`_seam_send_forward_recv_backward`; the consumer
        representative ships the gradient to every producer rank and all consumer
        ranks receive the next microbatch's features in the same exchange.
        """
        prod, cons = self._seam_producer, self._lm_layout
        feat = None
        for d in range(self._dp_size):
            producer_rep = prod.rank_at(d, prod.last_stage, 0, 0, 0)
            consumer_rep = cons.rank_at(d, 0, 0, 0, 0)
            if self._my_rank == consumer_rep:
                feat = exchange_objects(
                    grad, prod.stage_ranks(d, prod.last_stage), [producer_rep], self._device
                )[0]
            elif self._my_rank in cons.stage_ranks(d, 0):
                feat = exchange_objects(None, [], [producer_rep], self._device)[0]
        if isinstance(feat, torch.Tensor):
            feat.requires_grad_(True)
        return feat

    # ------------------------------------------------------------------
    # Neighbor communication (dispatch seam vs intra-mesh)
    # ------------------------------------------------------------------

    def _recv_forward(self) -> Any | None:
        if self.is_first_stage():
            return None
        if self._seam_on_recv():
            return self._seam_recv_forward()
        return self._lm_comm.recv_forward()

    def _send_forward(self, obj: Any) -> None:
        if self.is_last_stage():
            return
        if self._seam_on_send():
            self._seam_send_forward(obj)
            return
        self._lm_comm.send_forward(obj)

    def _recv_backward(self) -> Any | None:
        if self.is_last_stage():
            return None
        if self._seam_on_send():
            return self._seam_recv_backward()
        return self._lm_comm.recv_backward()

    def _send_backward(self, grad: Any) -> None:
        if self.is_first_stage():
            return
        if self._seam_on_recv():
            self._seam_send_backward(grad)
            return
        self._lm_comm.send_backward(grad)

    def _send_forward_recv_backward(self, obj: Any) -> Any | None:
        if self.is_last_stage():
            return None
        if self._seam_on_send():
            return self._seam_send_forward_recv_backward(obj)
        return self._lm_comm.send_forward_recv_backward(obj)

    def _send_backward_recv_forward(self, grad: Any) -> Any | None:
        if self.is_first_stage():
            return None
        if self._seam_on_recv():
            return self._seam_send_backward_recv_forward(grad)
        return self._lm_comm.send_backward_recv_forward(grad)


# ---------------------------------------------------------------------------
# 1F1B schedule
# ---------------------------------------------------------------------------

class OneForwardOneBackwardSchedule(BasePipelineSchedule):
    """One-forward-one-backward pipeline schedule over the global pipeline."""

    def step(
        self,
        microbatches: list[dict[str, torch.Tensor]],
        criterion: Callable[..., Any],
        optimizer: Optimizer | None = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        if self._idle:
            # No stage in this batch's plan (e.g. encoder ranks on a text-only
            # step): nothing to run, no transfers to anyone.
            return {"loss": None, "outputs": None}

        if not isinstance(microbatches, list):
            microbatches = [microbatches]
        num_microbatches = len(microbatches)
        self._device = _first_tensor_device(microbatches) or _default_device()

        num_warmup = min(self._num_stages - self._stage - 1, num_microbatches)
        num_steady = num_microbatches - num_warmup

        accum_loss: torch.Tensor | None = None
        if return_loss and self.is_last_stage():
            accum_loss = torch.zeros(1, device=self._device)
        outputs: list[Any] | None = (
            [] if return_outputs and self.is_last_stage() else None
        )

        input_objs: list[Any] = []
        output_objs: list[Any] = []
        mb_index = 0

        # Warmup.
        for _ in range(num_warmup):
            input_obj = self._recv_forward()
            output_obj = self._forward_step(
                microbatches[mb_index], input_obj, criterion,
                num_microbatches, accum_loss, outputs,
            )
            mb_index += 1
            self._send_forward(output_obj)
            input_objs.append(input_obj)
            output_objs.append(output_obj)

        if num_steady > 0:
            input_obj = self._recv_forward()

        # Steady state.
        for i in range(num_steady):
            last = i == num_steady - 1
            output_obj = self._forward_step(
                microbatches[mb_index], input_obj, criterion,
                num_microbatches, accum_loss, outputs,
            )
            mb_index += 1
            output_obj_grad = self._send_forward_recv_backward(output_obj)
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            input_obj_grad = self._backward_step(input_obj, output_obj, output_obj_grad)

            if last:
                self._send_backward(input_obj_grad)
            else:
                input_obj = self._send_backward_recv_forward(input_obj_grad)

        # Cooldown.
        for _ in range(num_warmup):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            output_obj_grad = self._recv_backward()
            input_obj_grad = self._backward_step(input_obj, output_obj, output_obj_grad)
            self._send_backward(input_obj_grad)

        return {
            "loss": accum_loss if return_loss else None,
            "outputs": _merge_batch(outputs) if outputs is not None else None,
        }
