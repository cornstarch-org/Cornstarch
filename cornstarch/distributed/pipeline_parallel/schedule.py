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
pipeline-stage boundary — a *seam*. Projected rows cross it through a DP-local,
variable-split all-to-all that maps the encoder and language-model CP layouts;
ordinary point-to-point transport remains in use for intra-model PP hops.

``TrainingSchedule``
    Abstract base: ``step(microbatches, criterion, optimizer)`` runs one
    forward-backward training step over a list of microbatches.

``BasePipelineSchedule``
    Builds, for this rank, its place in the global pipeline (which stage it owns,
    how it forwards, and how it communicates with its neighbors), and the shared
    forward/backward microbatch machinery. Subclasses implement the ordering.

Concrete orderings live in ``schedule_1f1b`` and ``schedule_zbpp``. They are
also lazily re-exported from this module for import compatibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from cornstarch.distributed.pipeline_parallel.p2p import (
    PipelineP2PCommunication,
)
from cornstarch.distributed.cross_mesh_routing import (
    CP_MODALITY_MASKS_KEY,
    CP_ROUTING_OFFSETS_KEY,
    CrossMeshGroup,
    CrossMeshRouter,
    SeamExchangeState,
    routing_offsets_from_batch,
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
        cross_mesh_groups: dict[tuple[int, int], list[CrossMeshGroup]] | None = None,
        routing_splitters: dict[int, tuple[Any, int]] | None = None,
    ) -> None:
        self._plan = plan
        self._output_future = output_future
        self._layouts = layouts
        self._dp_size = dp_size
        self._my_rank = dist.get_rank()
        self._device: torch.device | None = None
        self._seam_states: list[SeamExchangeState] = []
        self._routing_splitters = dict(routing_splitters or {})

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
            producer_module = self._encoder_nodes[0].params["module"]
            consumer_module = self._lm_node.params["module"]
            self._seam_producer_module_id = id(producer_module)
            self._seam_consumer_module_id = id(consumer_module)
            seam_groups = (cross_mesh_groups or {}).get(
                (self._seam_producer_module_id, self._seam_consumer_module_id)
            )
            if seam_groups is None:
                raise ValueError(
                    "A multimodal pipeline schedule requires cross-mesh seam "
                    "groups. Construct it with ParallelContext.create_schedule()."
                )
            self._seam_router = CrossMeshRouter(seam_groups)
            self._seam_modality = getattr(producer_module, "modality", None)
            if self._seam_modality is None:
                raise ValueError("A modality encoder seam must declare its modality name.")

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

    def _seam_spec(self, obj: torch.Tensor | None) -> tuple[int, torch.dtype]:
        hidden_size = int(self._lm_node.params["module"].hf_config.hidden_size)
        if obj is not None:
            if obj.shape[-1] != hidden_size:
                raise ValueError(
                    f"Projected modality hidden size {obj.shape[-1]} does not match "
                    f"language-model hidden size {hidden_size}."
                )
            return hidden_size, obj.dtype
        embedding = self._lm_node.params["module"].pre_decoder["embed_tokens"]
        parameter = next(embedding.parameters())
        return hidden_size, parameter.dtype

    def _routing_offsets(self, microbatch: dict, module_id: int, layout: MeshLayout):
        metadata = microbatch.get(CP_ROUTING_OFFSETS_KEY)
        if isinstance(metadata, dict) and module_id in metadata:
            return routing_offsets_from_batch(microbatch, module_id)
        if module_id == self._seam_producer_module_id:
            global_ids = microbatch.get("cp_global_input_ids", microbatch["input_ids"])
            counts = (global_ids == self._modality_token_ids[self._seam_modality]).sum(
                dim=1
            )
            modality_length = int(counts.max().item()) if counts.numel() else 0
            mask = (
                torch.arange(modality_length, device=global_ids.device).unsqueeze(0)
                < counts.unsqueeze(1)
            )
            splitter, cp_size = self._routing_splitters[module_id]
            if splitter is None:
                return (torch.arange(modality_length, dtype=torch.long),)
            return tuple(splitter.offsets_for_size(mask, cp_size))
        if layout.cp_size == 1:
            global_ids = microbatch.get("cp_global_input_ids", microbatch["input_ids"])
            return (torch.arange(global_ids.shape[1], dtype=torch.long),)
        raise ValueError(
            "Context-sharded modality routing requires per-microbatch offsets; "
            "build batches with ParallelContext.prepare_dataloader()."
        )

    def _seam_forward(
        self, obj: torch.Tensor | None, microbatch: dict
    ) -> torch.Tensor | None:
        global_ids = microbatch.get("cp_global_input_ids", microbatch.get("input_ids"))
        if not isinstance(global_ids, torch.Tensor):
            raise ValueError("Cross-mesh routing requires global input_ids metadata.")
        modality_masks = microbatch.get(CP_MODALITY_MASKS_KEY, {})
        if not isinstance(modality_masks, Mapping):
            raise ValueError("cp_modality_attention_masks must be a modality mapping.")
        source_attention_mask = modality_masks.get(self._seam_modality)
        if source_attention_mask is None:
            counts = (global_ids == self._modality_token_ids[self._seam_modality]).sum(
                dim=1
            )
            modality_length = int(counts.max().item()) if counts.numel() else 0
            source_attention_mask = (
                torch.arange(modality_length, device=global_ids.device).unsqueeze(0)
                < counts.unsqueeze(1)
            )
        hidden_size, dtype = self._seam_spec(obj)
        received, state = self._seam_router.forward(
            obj,
            global_input_ids=global_ids,
            token_id=self._modality_token_ids[self._seam_modality],
            source_attention_mask=source_attention_mask,
            source_offsets=self._routing_offsets(
                microbatch, self._seam_producer_module_id, self._seam_producer
            ),
            destination_offsets=self._routing_offsets(
                microbatch, self._seam_consumer_module_id, self._lm_layout
            ),
            hidden_size=hidden_size,
            dtype=dtype,
            device=self._device,
        )
        self._seam_states.append(state)
        if received is not None:
            received.requires_grad_(True)
        return received

    def _seam_backward(self, grad: torch.Tensor | None) -> torch.Tensor | None:
        if not self._seam_states:
            raise RuntimeError("Cross-mesh backward has no matching forward state.")
        state = self._seam_states.pop(0)
        hidden_size, dtype = self._seam_spec(grad)
        return self._seam_router.backward(
            grad,
            state,
            hidden_size=hidden_size,
            dtype=dtype,
            device=self._device,
        )

    def _seam_send_forward(self, obj: Any, microbatch: dict) -> None:
        self._seam_forward(obj, microbatch)

    def _seam_recv_forward(self, microbatch: dict) -> Any | None:
        return self._seam_forward(None, microbatch)

    def _seam_send_backward(self, grad: Any) -> None:
        self._seam_backward(grad)

    def _seam_recv_backward(self) -> Any | None:
        return self._seam_backward(None)

    def _seam_send_forward_recv_backward(
        self, obj: Any, microbatch: dict
    ) -> Any | None:
        self._seam_forward(obj, microbatch)
        return self._seam_backward(None)

    def _seam_send_backward_recv_forward(
        self, grad: Any, microbatch: dict
    ) -> Any | None:
        # Both sides issue collectives in forward-then-backward order. This is
        # the deterministic ordering required when adjacent 1F1B microbatches
        # overlap at a cross-mesh seam.
        feat = self._seam_forward(None, microbatch)
        self._seam_backward(grad)
        return feat

    # ------------------------------------------------------------------
    # Neighbor communication (dispatch seam vs intra-mesh)
    # ------------------------------------------------------------------

    def _recv_forward(self, microbatch: dict) -> Any | None:
        if self.is_first_stage():
            return None
        if self._seam_on_recv():
            return self._seam_recv_forward(microbatch)
        return self._lm_comm.recv_forward()

    def _send_forward(self, obj: Any, microbatch: dict) -> None:
        if self.is_last_stage():
            return
        if self._seam_on_send():
            self._seam_send_forward(obj, microbatch)
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

    def _send_forward_recv_backward(self, obj: Any, microbatch: dict) -> Any | None:
        if self.is_last_stage():
            return None
        if self._seam_on_send():
            return self._seam_send_forward_recv_backward(obj, microbatch)
        return self._lm_comm.send_forward_recv_backward(obj)

    def _send_backward_recv_forward(self, grad: Any, microbatch: dict) -> Any | None:
        if self.is_first_stage():
            return None
        if self._seam_on_recv():
            return self._seam_send_backward_recv_forward(grad, microbatch)
        return self._lm_comm.send_backward_recv_forward(grad)


def __getattr__(name: str) -> Any:
    """Lazily preserve concrete-schedule imports from this legacy module."""
    if name == "OneForwardOneBackwardSchedule":
        from cornstarch.distributed.pipeline_parallel.schedule_1f1b import (
            OneForwardOneBackwardSchedule,
        )

        return OneForwardOneBackwardSchedule
    if name == "ZeroBubblePipelineSchedule":
        from cornstarch.distributed.pipeline_parallel.schedule_zbpp import (
            ZeroBubblePipelineSchedule,
        )

        return ZeroBubblePipelineSchedule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
