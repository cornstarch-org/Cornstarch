import itertools
from collections import defaultdict

import numpy as np
import torch.distributed as dist

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)


class EncodersColocatedProcessGroupMesh(MultiModalProcessGroupMesh):
    """
    A helper class to manage the process group mesh.

    We use a ND-tuple to represent the process group mesh,
    and a ND-coordinate is to represent a process.

    EncodersColocatedProcessGroupMesh is a hybrid of ColossalAI ProcessGroupMesh
    and Cornstarch MultimodalProcessGroupMesh.
    Similar to ColossalAI ProcessGroupMesh, each stage is represented by a rank
    (ColossalAI MultimodalProcessGroupMesh assigns a rank per layer).
    Similar to Cornstarch MultimodalProcessGroupMesh, EncodersColocatedProcessGroupMesh
    supports heterogeneous ranks between encoder and LLM: LLM and encoder may have different
    tensor parallel degree or sequence parallel degree.

    Args:
        encoder_parallel_size (tuple[int, int]): The pipeline parallel degree and tensor parallel
            degree of the encoder.
        llm_parallel_size (tuple[int, int, int]): The pipeline parallel degree, tensor parallel
            degree and sequence parallel degree of the LLM.

    Currently Sequence parallelism for encoders is not supported.
    """

    def __init__(
        self,
        encoder_templates: dict[PipelineTemplate, int | tuple[int, int]],
        llm_template: tuple[PipelineTemplate, int, int],
    ) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."
        assert isinstance(encoder_templates, dict), "encoder_templates must be a dict."
        assert len(encoder_templates) > 0, "encoder_templates must not be empty."

        for encoder_template, parallel_sizes in encoder_templates.items():
            if isinstance(parallel_sizes, int):
                encoder_templates[encoder_template] = (parallel_sizes, 1)

        first_encoder_template = next(iter(encoder_templates.keys()))
        for template, parallel_size in encoder_templates.items():
            assert (
                parallel_size == encoder_templates[first_encoder_template]
            ), "All encoder templates must have the same tensor/context parallel degree."
            assert (
                template.num_stages == first_encoder_template.num_stages
            ), "All encoder templates must have the same number of stages."

        self.encoder_templates = encoder_templates
        self.llm_template = llm_template

        # Context parallelism for encoders is not supported. SP for encoders == 1.
        encoder_pp_size, (encoder_tp_size, encoder_sp_size) = (
            first_encoder_template.num_stages,
            encoder_templates[first_encoder_template],
        )
        llm_pp_size, llm_tp_size, llm_sp_size = (
            llm_template[0].num_stages,
            llm_template[1],
            llm_template[2],
        )

        num_ranks_in_model = (
            encoder_pp_size * encoder_tp_size * encoder_sp_size
            + llm_pp_size * llm_tp_size * llm_sp_size
        )
        assert (
            dist.get_world_size() % num_ranks_in_model == 0
        ), f"The world size {dist.get_world_size()} must be divisible by num_ranks per replica {num_ranks_in_model}."
        dp_size = dist.get_world_size() // num_ranks_in_model

        max_tp_size = max(encoder_tp_size, llm_tp_size)
        max_sp_size = max(encoder_sp_size, llm_sp_size)
        assert all(max_tp_size % tp == 0 for tp in [encoder_tp_size, llm_tp_size]), (
            f"TP size {max_tp_size} must be divisible by encoder TP size {encoder_tp_size} "
            f"and LLM TP size {llm_tp_size}."
        )
        assert all(max_sp_size % sp == 0 for sp in [encoder_sp_size, llm_sp_size]), (
            f"SP size {max_sp_size} must be divisible by encoder SP size {encoder_sp_size} "
            f"and LLM SP size {llm_sp_size}."
        )

        meshes: list[list[list[int]]] = []
        rank_index = 0
        modal_to_ranks: dict[PipelineTemplate, list[int]] = defaultdict(list)

        iterables: list[tuple[PipelineTemplate, int, int]] = [
            (first_encoder_template, encoder_tp_size, encoder_sp_size),
            llm_template,
        ]

        for modal, tp_size, sp_size in iterables:
            for _ in range(modal.num_stages):
                stage_mesh = []
                for _ in range(dp_size):
                    tp_mesh = []
                    copy_per_same_ranks = max_sp_size // sp_size

                    for _ in range(sp_size):
                        ranks = list(
                            itertools.chain.from_iterable(
                                [
                                    [i] * (max_tp_size // tp_size)
                                    for i in range(rank_index, rank_index + tp_size)
                                ]
                            )
                        )
                        rank_index += tp_size
                        tp_mesh.extend([ranks for _ in range(copy_per_same_ranks)])

                    stage_mesh.append(tp_mesh)

                    if modal == first_encoder_template:
                        for encoder in encoder_templates.keys():
                            modal_to_ranks[encoder].extend(
                                list(itertools.chain(*tp_mesh))
                            )
                    else:
                        assert modal == llm_template[0]
                        modal_to_ranks[modal].extend(list(itertools.chain(*tp_mesh)))
                meshes.append(stage_mesh)

        self._rank = dist.get_rank()
        self._mesh = np.array(meshes)
        self._shape = self._mesh.shape

        self._coords = MultiModalProcessGroupMesh.unravel(self._rank, self._mesh)
        self._ranks_to_group: dict[tuple[int, ...], dist.ProcessGroup] = {}
        self._group_to_ranks: dict[dist.ProcessGroup, tuple[int, ...]] = {}

        self.modal_to_ranks = {
            modal: list(set(ranks)) for modal, ranks in modal_to_ranks.items()
        }
