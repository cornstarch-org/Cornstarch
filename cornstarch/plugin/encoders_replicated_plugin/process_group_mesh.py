import itertools
from collections import defaultdict

import numpy as np
import torch.distributed as dist
from transformers.utils import logging

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)

logger = logging.get_logger(__name__)


class EncodersReplicatedProcessGroupMesh(MultiModalProcessGroupMesh):
    """
    Args:
        llm_template (tuple[int, int, int]): The pipeline parallel degree, tensor parallel
            degree and sequence parallel degree of the LLM.

    Currently Sequence parallelism for encoders is not supported.
    """

    def __init__(
        self,
        llm_template: tuple[PipelineTemplate, int, int],
    ):
        assert dist.is_initialized(), "Please initialize torch.distributed first."
        assert llm_template is not None, "llm_template must be provided."

        logger.warning_once(
            "Tensor parallel and pipeline parallel configuration for all encoders "
            "will be ignored in EncodersReplicated."
        )

        self.llm_template = llm_template

        llm_pp_size, llm_tp_size, llm_sp_size = (
            llm_template[0].num_stages,
            llm_template[1],
            llm_template[2],
        )

        num_ranks_in_model = llm_pp_size * llm_tp_size * llm_sp_size
        assert (
            dist.get_world_size() % num_ranks_in_model == 0
        ), f"World size {dist.get_world_size()} is not divisible by num_ranks per replica {num_ranks_in_model}."
        dp_size = dist.get_world_size() // num_ranks_in_model

        meshes: list[list[list[int]]] = []
        rank_index = 0
        modal_to_ranks: dict[PipelineTemplate, list[int]] = defaultdict(list)

        # Encoders are replicated for every LLM stages.
        for _ in range(llm_pp_size):
            stage_mesh = []
            for _ in range(dp_size):
                tp_mesh = []
                for _ in range(llm_sp_size):
                    ranks = [i for i in range(rank_index, rank_index + llm_tp_size)]
                    rank_index += llm_tp_size
                    tp_mesh.append(ranks)
                stage_mesh.append(tp_mesh)
                modal_to_ranks[llm_template[0]].extend(list(itertools.chain(*tp_mesh)))
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
