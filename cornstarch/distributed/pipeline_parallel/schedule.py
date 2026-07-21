"""Training schedules for pipeline-parallel distributed execution.

A schedule exists **only when pipeline parallelism is used**. Without pipeline
parallelism the modules are co-located on every rank and the training loop runs
the execution plan directly (``output_future.execute()`` + ``backward()``),
accumulating gradients over the user's ``collate_fn`` microbatch list — no
schedule is involved.

When pipeline parallelism is used the modules are disaggregated onto disjoint
rank ranges and form a pipeline:

- one ordinary or fused **modality encoder pipeline** leads the graph; every
  fused child spans the same encoder stages and moves in registry order, feeding
- the independently configured **language-model pipeline** (``merge`` runs on
  its first stage).

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

``OneForwardOneBackwardSchedule``
    The 1F1B ordering (warmup / steady / cooldown).
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
    _resolve_optional_modality_inputs,
    _resolve_value,
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

    The global pipeline is ``[encoder PP stages..., language-model PP stages...]``.
    This rank owns exactly one global stage because the module meshes are disjoint.
    For a fused producer, every registered child runs its corresponding local
    partition on every encoder stage. The base class resolves that placement,
    runs this rank's stage
    forward/backward per microbatch, and moves activations to its neighbors —
    using ordinary pipeline P2P for intra-mesh hops and the cross-mesh *seam*
    transport once per modality at the encoder→language-model boundary. Subclasses choose the
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
        self._seam_states: list[tuple[tuple[str, SeamExchangeState], ...]] = []
        self._routing_splitters = dict(routing_splitters or {})

        nodes = plan._topological_nodes(output_future.name)
        self._encoder_nodes = [
            node
            for node in nodes
            if node.kind in {"run_modality_encoder", "run_fused_modality_encoder"}
        ]
        if len(self._encoder_nodes) > 1:
            raise ValueError(
                "Pipeline schedules accept one encoder producer. Compose multiple "
                "modalities with CornstarchFusedModalityEncoder."
            )
        self._encoder_node = self._encoder_nodes[0] if self._encoder_nodes else None
        self._merge_node = next(
            (node for node in nodes if node.kind == "merge_modality_encoder_outputs"),
            None,
        )
        self._lm_node = next(node for node in nodes if node.kind == "run_language_model")
        self._modality_token_ids: dict[str, int] = (
            dict(self._merge_node.params.get("modality_token_ids", {}))
            if self._merge_node is not None
            else {}
        )

        self._lm_layout = layouts[id(self._lm_node.params["module"])]
        self._has_encoder = self._encoder_node is not None
        self._num_encoder_stages = 0
        if self._has_encoder:
            producer_module = self._encoder_node.params["module"]
            self._encoder_layout = layouts[id(producer_module)]
            self._num_encoder_stages = self._encoder_layout.num_pp_stages
        self._num_stages = self._num_encoder_stages + self._lm_layout.num_pp_stages

        # Resolve this rank's one local role in the disaggregated global pipeline.
        self._role: str | None = None
        if self._has_encoder and self._my_rank in self._encoder_layout.rankset:
            self._role = "encoder"
            self._encoder_mesh = meshes[id(self._encoder_node.params["module"])]
            self._encoder_comm = PipelineP2PCommunication(self._encoder_mesh)
            self._encoder_stage = self._encoder_mesh.stage
            self._stage = self._encoder_stage
        elif self._my_rank in self._lm_layout.rankset:
            self._role = "llm"
            self._lm_mesh = meshes[id(self._lm_node.params["module"])]
            self._lm_comm = PipelineP2PCommunication(self._lm_mesh)
            self._lm_stage = self._lm_mesh.stage
            self._stage = self._num_encoder_stages + self._lm_stage

        # A rank with no node in this batch's plan idles (for example encoder
        # ranks on a text-only step).
        self._idle = self._role is None

        if self._has_encoder:
            producer_module = self._encoder_node.params["module"]
            consumer_module = self._lm_node.params["module"]
            self._seam_producer = self._encoder_layout
            self._encoder_output_name = self._encoder_node.name
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
            if self._encoder_node.kind == "run_fused_modality_encoder":
                provided = self._encoder_node.params["inputs"]
                self._seam_modalities = tuple(
                    modality
                    for modality in producer_module.modalities
                    if modality in provided
                )
            else:
                modality = getattr(producer_module, "modality", None)
                if modality is None:
                    raise ValueError(
                        "A modality encoder seam must declare its modality name."
                    )
                self._seam_modalities = (modality,)
            missing_tokens = set(self._seam_modalities) - set(self._modality_token_ids)
            if missing_tokens:
                raise ValueError(
                    "Missing modality token IDs for fused seam outputs: "
                    f"{sorted(missing_tokens)}"
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
        """This rank sends across the seam only at the encoder's final stage."""
        return (
            self._role == "encoder"
            and self._encoder_stage == self._num_encoder_stages - 1
        )

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
            values = dict(microbatch)
            if self._encoder_stage == 0:
                output = CornstarchExecutionPlan._execute_node(
                    self._encoder_node, values
                )
            elif self._encoder_node.kind == "run_fused_modality_encoder":
                if not isinstance(input_obj, Mapping):
                    raise TypeError(
                        "A fused encoder PP stage expects a modality activation mapping."
                    )
                stage_inputs: dict[str, dict[str, Any]] = {}
                for modality, declared in self._encoder_node.params["inputs"].items():
                    if modality not in input_obj:
                        continue
                    kwargs = _resolve_optional_modality_inputs(declared, values)
                    if kwargs is None:
                        raise ValueError(
                            f"{modality!r} has a pipeline activation but its "
                            "optional source inputs are absent."
                        )
                    activation = input_obj[modality]
                    if isinstance(activation, Mapping):
                        kwargs.update(activation)
                    else:
                        kwargs["hidden_states"] = activation
                    stage_inputs[modality] = kwargs
                output = self._encoder_node.params["module"](inputs=stage_inputs)
            else:
                kwargs = {
                    key: _resolve_value(value, values)
                    for key, value in self._encoder_node.params["inputs"].items()
                }
                if isinstance(input_obj, Mapping):
                    kwargs.update(input_obj)
                elif input_obj is not None:
                    kwargs["hidden_states"] = input_obj
                output = self._encoder_node.params["module"](**kwargs)

            if self._encoder_node.kind == "run_fused_modality_encoder":
                return {
                    modality: _first_output_tensor(modality_output)
                    for modality, modality_output in output.items()
                }
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
                if not tensors:
                    pass
                elif len(tensors) == 1:
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
        device_type = self._device.type
        if torch.is_autocast_enabled(device_type):
            return hidden_size, torch.get_autocast_dtype(device_type)
        embedding = self._lm_node.params["module"].pre_decoder["embed_tokens"]
        parameter = next(embedding.parameters())
        return hidden_size, parameter.dtype

    def _routing_offsets(
        self,
        microbatch: dict,
        module_id: int,
        layout: MeshLayout,
        modality: str,
    ):
        metadata = microbatch.get(CP_ROUTING_OFFSETS_KEY)
        if isinstance(metadata, Mapping) and module_id in metadata:
            module_offsets = metadata[module_id]
            if isinstance(module_offsets, Mapping):
                if modality not in module_offsets:
                    raise ValueError(
                        f"Missing CP routing offsets for fused modality {modality!r}."
                    )
                return module_offsets[modality]
            return routing_offsets_from_batch(microbatch, module_id)
        if module_id == self._seam_producer_module_id:
            global_ids = microbatch.get("cp_global_input_ids", microbatch["input_ids"])
            counts = (global_ids == self._modality_token_ids[modality]).sum(dim=1)
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

    def _seam_forward(self, obj: Any | None, microbatch: dict) -> Any | None:
        global_ids = microbatch.get("cp_global_input_ids", microbatch.get("input_ids"))
        if not isinstance(global_ids, torch.Tensor):
            raise ValueError("Cross-mesh routing requires global input_ids metadata.")
        modality_masks = microbatch.get(CP_MODALITY_MASKS_KEY, {})
        if not isinstance(modality_masks, Mapping):
            raise ValueError("cp_modality_attention_masks must be a modality mapping.")

        fused = len(self._seam_modalities) > 1 or (
            self._encoder_node.kind == "run_fused_modality_encoder"
        )
        if obj is not None and fused and not isinstance(obj, Mapping):
            raise TypeError("A fused seam sender must provide a modality mapping.")
        active_modalities = tuple(
            modality
            for modality in self._seam_modalities
            if bool((global_ids == self._modality_token_ids[modality]).any().item())
        )
        if isinstance(obj, Mapping) and set(obj) != set(active_modalities):
            raise ValueError(
                "Fused encoder outputs must exactly match modalities present in "
                f"the language microbatch: outputs={sorted(obj)}, "
                f"present={sorted(active_modalities)}"
            )
        received_by_modality: dict[str, torch.Tensor] = {}
        forward_states: list[tuple[str, SeamExchangeState]] = []
        for modality in active_modalities:
            feature = obj.get(modality) if isinstance(obj, Mapping) else obj
            source_attention_mask = modality_masks.get(modality)
            if source_attention_mask is None:
                counts = (global_ids == self._modality_token_ids[modality]).sum(dim=1)
                modality_length = int(counts.max().item()) if counts.numel() else 0
                source_attention_mask = (
                    torch.arange(modality_length, device=global_ids.device).unsqueeze(0)
                    < counts.unsqueeze(1)
                )
            hidden_size, dtype = self._seam_spec(feature)
            received, state = self._seam_router.forward(
                feature,
                global_input_ids=global_ids,
                token_id=self._modality_token_ids[modality],
                source_attention_mask=source_attention_mask,
                source_offsets=self._routing_offsets(
                    microbatch,
                    self._seam_producer_module_id,
                    self._seam_producer,
                    modality,
                ),
                destination_offsets=self._routing_offsets(
                    microbatch,
                    self._seam_consumer_module_id,
                    self._lm_layout,
                    modality,
                ),
                hidden_size=hidden_size,
                dtype=dtype,
                device=self._device,
            )
            forward_states.append((modality, state))
            if received is not None:
                received.requires_grad_(True)
                received_by_modality[modality] = received

        self._seam_states.append(tuple(forward_states))
        if fused:
            return received_by_modality
        return received_by_modality.get(self._seam_modalities[0])

    def _seam_backward(self, grad: Any | None) -> Any | None:
        fused = len(self._seam_modalities) > 1 or (
            self._encoder_node.kind == "run_fused_modality_encoder"
        )
        if grad is not None and fused and not isinstance(grad, Mapping):
            raise TypeError("A fused seam receiver must provide modality gradients.")
        if not self._seam_states:
            raise RuntimeError("Cross-mesh backward has no matching forward state.")
        forward_states = self._seam_states.pop(0)
        returned_by_modality: dict[str, torch.Tensor] = {}
        for modality, state in forward_states:
            modality_grad = grad.get(modality) if isinstance(grad, Mapping) else grad
            hidden_size, dtype = self._seam_spec(modality_grad)
            returned = self._seam_router.backward(
                modality_grad,
                state,
                hidden_size=hidden_size,
                dtype=dtype,
                device=self._device,
            )
            if returned is not None:
                returned_by_modality[modality] = returned

        if fused:
            return returned_by_modality
        return returned_by_modality.get(self._seam_modalities[0])

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
        if self._role == "encoder":
            return self._encoder_comm.recv_forward()
        return self._lm_comm.recv_forward()

    def _send_forward(self, obj: Any, microbatch: dict) -> None:
        if self.is_last_stage():
            return
        if self._seam_on_send():
            self._seam_send_forward(obj, microbatch)
            return
        if self._role == "encoder":
            self._encoder_comm.send_forward(obj)
            return
        self._lm_comm.send_forward(obj)

    def _recv_backward(self) -> Any | None:
        if self.is_last_stage():
            return None
        if self._seam_on_send():
            return self._seam_recv_backward()
        if self._role == "encoder":
            return self._encoder_comm.recv_backward()
        return self._lm_comm.recv_backward()

    def _send_backward(self, grad: Any) -> None:
        if self.is_first_stage():
            return
        if self._seam_on_recv():
            self._seam_send_backward(grad)
            return
        if self._role == "encoder":
            self._encoder_comm.send_backward(grad)
            return
        self._lm_comm.send_backward(grad)

    def _send_forward_recv_backward(self, obj: Any, microbatch: dict) -> Any | None:
        if self.is_last_stage():
            return None
        if self._seam_on_send():
            return self._seam_send_forward_recv_backward(obj, microbatch)
        if self._role == "encoder":
            return self._encoder_comm.send_forward_recv_backward(obj)
        return self._lm_comm.send_forward_recv_backward(obj)

    def _send_backward_recv_forward(self, grad: Any, microbatch: dict) -> Any | None:
        if self.is_first_stage():
            return None
        if self._seam_on_recv():
            return self._seam_send_backward_recv_forward(grad, microbatch)
        if self._role == "encoder":
            return self._encoder_comm.send_backward_recv_forward(grad)
        return self._lm_comm.send_backward_recv_forward(grad)


# ---------------------------------------------------------------------------
# 1F1B schedule
# ---------------------------------------------------------------------------

class OneForwardOneBackwardSchedule(BasePipelineSchedule):
    """One-forward-one-backward pipeline schedule over the global pipeline."""

    def extra_warmup_microbatches(
        self,
        microbatches: list[dict[str, torch.Tensor]],
        required_warmup: int,
    ) -> int:
        """Return coordinated eager forwards beyond the deadlock-safe minimum.

        Subclasses may override this public extension hook. The returned value
        must be identical on every active pipeline stage (coordinate it with a
        collective when it depends on rank-local memory); negative values are
        rejected and the total is capped by the iteration's microbatch count.
        """
        return 0

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

        required_warmup = min(
            self._num_stages - self._stage - 1, num_microbatches
        )
        extra_warmup = self.extra_warmup_microbatches(
            microbatches, required_warmup
        )
        if not isinstance(extra_warmup, int) or extra_warmup < 0:
            raise ValueError("extra_warmup_microbatches must return a nonnegative int")
        num_warmup = min(required_warmup + extra_warmup, num_microbatches)
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
            active_microbatch = microbatches[mb_index]
            input_obj = self._recv_forward(active_microbatch)
            output_obj = self._forward_step(
                active_microbatch, input_obj, criterion,
                num_microbatches, accum_loss, outputs,
            )
            mb_index += 1
            self._send_forward(output_obj, active_microbatch)
            input_objs.append(input_obj)
            output_objs.append(output_obj)

        if num_steady > 0:
            input_obj = self._recv_forward(microbatches[mb_index])

        # Steady state.
        for i in range(num_steady):
            last = i == num_steady - 1
            active_microbatch = microbatches[mb_index]
            output_obj = self._forward_step(
                active_microbatch, input_obj, criterion,
                num_microbatches, accum_loss, outputs,
            )
            mb_index += 1
            output_obj_grad = self._send_forward_recv_backward(
                output_obj, active_microbatch
            )
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            input_obj_grad = self._backward_step(input_obj, output_obj, output_obj_grad)

            if last:
                self._send_backward(input_obj_grad)
            else:
                input_obj = self._send_backward_recv_forward(
                    input_obj_grad, microbatches[mb_index]
                )

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
