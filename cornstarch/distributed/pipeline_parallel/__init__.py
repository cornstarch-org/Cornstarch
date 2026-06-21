"""Pipeline parallelism: P2P communication, forward spec wrapper, and 1F1B schedule."""
from __future__ import annotations

import torch.nn as nn

from cornstarch.distributed.pipeline_parallel.forward_spec_wrapper import (
    PipelineParallelForwardSpec,
)
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.models.model_base import CornstarchModelBase


def apply_pipeline_parallel(
    module: CornstarchModelBase,
    mesh: ModalProcessGroupMesh,
) -> None:
    """Distribute a model across pipeline stages.

    Slices the repeated layers to this rank's stage and wraps the module's
    ``forward_spec`` for stage-aware execution: the wrapper makes
    ``embed_inputs`` pass through received hidden states on non-first stages and
    bypasses ``finalize_hidden_states`` / ``build_output`` on non-last stages.

    The ``pre_*`` and ``post_*`` sections are intentionally **kept intact** (not
    replaced with ``nn.Identity``).  Their token-embedding and head modules are
    simply never invoked off the boundary stages because the forward spec gates
    them, while helper modules a spec needs on *every* stage — such as a rotary
    position embedding living in ``pre_decoder`` — must stay reachable.  The
    memory of the unused boundary modules is small relative to the repeated
    layers that PP actually shards.

    Must be called before ``module.materialize()``.
    """
    _, layers_name, _ = module._section_names()

    layers = getattr(module, layers_name)
    total_layers = len(layers)
    start, end = mesh.distribute_layers(total_layers)
    setattr(module, layers_name, nn.ModuleList(list(layers)[start:end]))

    module.forward_spec = PipelineParallelForwardSpec(
        module.forward_spec, mesh, layer_offset=start
    )
