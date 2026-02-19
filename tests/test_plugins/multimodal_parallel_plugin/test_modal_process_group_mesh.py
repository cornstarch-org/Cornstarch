import functools
from collections import defaultdict

import pytest
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)

from ..common import (
    encoder1_template,
    encoder2_template,
    llm_template_2stages,
    llm_template_4stages,
)


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_init_process_group_mesh
#
# Each parametrize entry now provides:
#   expected_modal_meshes: dict[str, list]  keyed by modal model_name
#   expected_ranks:        dict[PipelineTemplate, list[int]]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_modal_meshes, expected_ranks",
    [
        # ------------------------------------------------------------------
        # Case 1: encoder TP=2 SP=1, LLM TP=4 SP=1, DP=2
        # encoder mesh [2,2,1,2]; LLM mesh [2,2,1,4]
        # ------------------------------------------------------------------
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            {
                "encoder1": [
                    [[[0, 1]], [[2, 3]]],
                    [[[4, 5]], [[6, 7]]],
                ],
                "llm": [
                    [[[8, 9, 10, 11]], [[12, 13, 14, 15]]],
                    [[[16, 17, 18, 19]], [[20, 21, 22, 23]]],
                ],
            },
            {
                encoder1_template: list(range(0, 8)),
                llm_template_2stages: list(range(8, 24)),
            },
        ),
        # ------------------------------------------------------------------
        # Case 2: encoder TP=2 SP=2, LLM TP=2 SP=2, DP=1
        # encoder mesh [2,1,2,2]; LLM mesh [2,1,2,2]
        # ------------------------------------------------------------------
        (
            16,
            {encoder1_template: (2, 2)},
            (llm_template_2stages, 2, 2),
            {
                "encoder1": [
                    [[[0, 1], [2, 3]]],
                    [[[4, 5], [6, 7]]],
                ],
                "llm": [
                    [[[8, 9], [10, 11]]],
                    [[[12, 13], [14, 15]]],
                ],
            },
            {
                encoder1_template: list(range(0, 8)),
                llm_template_2stages: list(range(8, 16)),
            },
        ),
        # ------------------------------------------------------------------
        # Case 3: encoder1 TP=2 SP=1, encoder2 TP=2 SP=1, LLM TP=4 SP=1, DP=1
        # encoder1 mesh [2,1,1,2]; encoder2 mesh [3,1,1,2]; LLM mesh [2,1,1,4]
        # ------------------------------------------------------------------
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {
                "encoder1": [
                    [[[0, 1]]],
                    [[[2, 3]]],
                ],
                "encoder2": [
                    [[[4, 5]]],
                    [[[6, 7]]],
                    [[[8, 9]]],
                ],
                "llm": [
                    [[[10, 11, 12, 13]]],
                    [[[14, 15, 16, 17]]],
                ],
            },
            {
                encoder1_template: list(range(0, 4)),
                encoder2_template: list(range(4, 10)),
                llm_template_2stages: list(range(10, 18)),
            },
        ),
        # ------------------------------------------------------------------
        # Case 4: encoder2 TP=4 SP=1, LLM TP=4 SP=1, DP=3
        # encoder2 mesh [3,3,1,4]; LLM mesh [4,3,1,4]
        # ------------------------------------------------------------------
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            {
                "encoder2": [
                    [[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9, 10, 11]]],
                    [[[12, 13, 14, 15]], [[16, 17, 18, 19]], [[20, 21, 22, 23]]],
                    [[[24, 25, 26, 27]], [[28, 29, 30, 31]], [[32, 33, 34, 35]]],
                ],
                "llm": [
                    [[[36, 37, 38, 39]], [[40, 41, 42, 43]], [[44, 45, 46, 47]]],
                    [[[48, 49, 50, 51]], [[52, 53, 54, 55]], [[56, 57, 58, 59]]],
                    [[[60, 61, 62, 63]], [[64, 65, 66, 67]], [[68, 69, 70, 71]]],
                    [[[72, 73, 74, 75]], [[76, 77, 78, 79]], [[80, 81, 82, 83]]],
                ],
            },
            {
                encoder2_template: list(range(0, 36)),
                llm_template_4stages: list(range(36, 84)),
            },
        ),
        # ------------------------------------------------------------------
        # Case 5: encoder TP=2 SP=1, LLM TP=4 SP=2, DP=2
        # encoder mesh [2,2,1,2]; LLM mesh [2,2,2,4]
        # ------------------------------------------------------------------
        (
            40,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            {
                "encoder1": [
                    [[[0, 1]], [[2, 3]]],
                    [[[4, 5]], [[6, 7]]],
                ],
                "llm": [
                    [[[8, 9, 10, 11], [12, 13, 14, 15]],
                     [[16, 17, 18, 19], [20, 21, 22, 23]]],
                    [[[24, 25, 26, 27], [28, 29, 30, 31]],
                     [[32, 33, 34, 35], [36, 37, 38, 39]]],
                ],
            },
            {
                encoder1_template: list(range(0, 8)),
                llm_template_2stages: list(range(8, 40)),
            },
        ),
        # ------------------------------------------------------------------
        # Case 6: encoder1 TP=2 SP=1, encoder2 TP=2 SP=1, LLM TP=4 SP=4, DP=1
        # encoder1 mesh [2,1,1,2]; encoder2 mesh [3,1,1,2]; LLM mesh [2,1,4,4]
        # ------------------------------------------------------------------
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            {
                "encoder1": [
                    [[[0, 1]]],
                    [[[2, 3]]],
                ],
                "encoder2": [
                    [[[4, 5]]],
                    [[[6, 7]]],
                    [[[8, 9]]],
                ],
                "llm": [
                    [[[10, 11, 12, 13],
                      [14, 15, 16, 17],
                      [18, 19, 20, 21],
                      [22, 23, 24, 25]]],
                    [[[26, 27, 28, 29],
                      [30, 31, 32, 33],
                      [34, 35, 36, 37],
                      [38, 39, 40, 41]]],
                ],
            },
            {
                encoder1_template: list(range(0, 4)),
                encoder2_template: list(range(4, 10)),
                llm_template_2stages: list(range(10, 42)),
            },
        ),
        # ------------------------------------------------------------------
        # Case 7: encoder2 TP=4 SP=1, LLM TP=2 SP=4, DP=3
        # encoder2 mesh [3,3,1,4]; LLM mesh [4,3,4,2]
        # ------------------------------------------------------------------
        (
            132,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            {
                "encoder2": [
                    [[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9, 10, 11]]],
                    [[[12, 13, 14, 15]], [[16, 17, 18, 19]], [[20, 21, 22, 23]]],
                    [[[24, 25, 26, 27]], [[28, 29, 30, 31]], [[32, 33, 34, 35]]],
                ],
                "llm": [
                    [[[36, 37], [38, 39], [40, 41], [42, 43]],
                     [[44, 45], [46, 47], [48, 49], [50, 51]],
                     [[52, 53], [54, 55], [56, 57], [58, 59]]],
                    [[[60, 61], [62, 63], [64, 65], [66, 67]],
                     [[68, 69], [70, 71], [72, 73], [74, 75]],
                     [[76, 77], [78, 79], [80, 81], [82, 83]]],
                    [[[84, 85], [86, 87], [88, 89], [90, 91]],
                     [[92, 93], [94, 95], [96, 97], [98, 99]],
                     [[100, 101], [102, 103], [104, 105], [106, 107]]],
                    [[[108, 109], [110, 111], [112, 113], [114, 115]],
                     [[116, 117], [118, 119], [120, 121], [122, 123]],
                     [[124, 125], [126, 127], [128, 129], [130, 131]]],
                ],
            },
            {
                encoder2_template: list(range(0, 36)),
                llm_template_4stages: list(range(36, 132)),
            },
        ),
        # ------------------------------------------------------------------
        # Case A (new): encoder TP=4 SP=1, LLM TP=2 SP=1 (only TP different,
        #               encoder TP > LLM TP), DP=1
        # encoder mesh [2,1,1,4]; LLM mesh [2,1,1,2]
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 4},
            (llm_template_2stages, 2, 1),
            {
                "encoder1": [
                    [[[0, 1, 2, 3]]],
                    [[[4, 5, 6, 7]]],
                ],
                "llm": [
                    [[[8, 9]]],
                    [[[10, 11]]],
                ],
            },
            {
                encoder1_template: list(range(0, 8)),
                llm_template_2stages: list(range(8, 12)),
            },
        ),
        # ------------------------------------------------------------------
        # Case B (new): encoder TP=2 SP=1, LLM TP=2 SP=2 (only SP different,
        #               encoder SP < LLM SP), DP=1
        # encoder mesh [2,1,1,2]; LLM mesh [2,1,2,2]
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 2),
            {
                "encoder1": [
                    [[[0, 1]]],
                    [[[2, 3]]],
                ],
                "llm": [
                    [[[4, 5], [6, 7]]],
                    [[[8, 9], [10, 11]]],
                ],
            },
            {
                encoder1_template: list(range(0, 4)),
                llm_template_2stages: list(range(4, 12)),
            },
        ),
        # ------------------------------------------------------------------
        # Case C (new): encoder TP=2 SP=2, LLM TP=2 SP=1 (only SP different,
        #               encoder SP > LLM SP), DP=1
        # encoder mesh [2,1,2,2]; LLM mesh [2,1,1,2]
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: (2, 2)},
            (llm_template_2stages, 2, 1),
            {
                "encoder1": [
                    [[[0, 1], [2, 3]]],
                    [[[4, 5], [6, 7]]],
                ],
                "llm": [
                    [[[8, 9]]],
                    [[[10, 11]]],
                ],
            },
            {
                encoder1_template: list(range(0, 8)),
                llm_template_2stages: list(range(8, 12)),
            },
        ),
        # ------------------------------------------------------------------
        # Case D (new): encoder TP=4 SP=2, LLM TP=2 SP=1 (both different,
        #               encoder TP > LLM TP AND encoder SP > LLM SP), DP=1
        # encoder mesh [2,1,2,4]; LLM mesh [2,1,1,2]
        # ------------------------------------------------------------------
        (
            20,
            {encoder1_template: (4, 2)},
            (llm_template_2stages, 2, 1),
            {
                "encoder1": [
                    [[[0, 1, 2, 3], [4, 5, 6, 7]]],
                    [[[8, 9, 10, 11], [12, 13, 14, 15]]],
                ],
                "llm": [
                    [[[16, 17]]],
                    [[[18, 19]]],
                ],
            },
            {
                encoder1_template: list(range(0, 16)),
                llm_template_2stages: list(range(16, 20)),
            },
        ),
    ],
)
def test_init_process_group_mesh(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_modal_meshes: dict[str, list],
    expected_ranks: dict[PipelineTemplate, list[int]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
        assert (
            mesh.mesh == expected_modal_meshes[mesh.my_modal.model_name]
        ).all(), (
            f"rank {rank} modal {mesh.my_modal.model_name}: mesh mismatch.\n"
            f"  expected: {expected_modal_meshes[mesh.my_modal.model_name]}\n"
            f"  got:      {mesh.mesh.tolist()}"
        )
        assert mesh.modal_to_ranks == expected_ranks

        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_get_group_along_axis
#
# expected_group_ranks[axis] is now a list of intra-modal rank tuples.
# Only the PP axis changes versus the old tests; DP/TP/SP remain the same.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_group_ranks",
    (
        # ------------------------------------------------------------------
        # encoder TP=2 SP=1, LLM TP=4 SP=1, DP=2
        # ------------------------------------------------------------------
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder intra-modal PP groups (2 stages, DP=2, TP=2)
                    (0, 4), (1, 5), (2, 6), (3, 7),
                    # LLM intra-modal PP groups (2 stages, DP=2, TP=4)
                    (8, 16), (9, 17), (10, 18), (11, 19),
                    (12, 20), (13, 21), (14, 22), (15, 23),
                ],
                MultiModalProcessGroupMesh.dp_axis: [
                    (0, 2), (1, 3), (4, 6), (5, 7),
                    (8, 12), (9, 13), (10, 14), (11, 15),
                    (16, 20), (17, 21), (18, 22), (19, 23),
                ],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1), (2, 3), (4, 5), (6, 7),
                    (8, 9, 10, 11), (12, 13, 14, 15),
                    (16, 17, 18, 19), (20, 21, 22, 23),
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder1 TP=2 SP=1, encoder2 TP=2 SP=1, LLM TP=4 SP=1, DP=1
        # encoder2 has PP=3 so its full-pipeline PP groups span all 3 stages
        # ------------------------------------------------------------------
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder1 (2 stages, DP=1, TP=2): one group per (dp,tp)
                    (0, 2), (1, 3),
                    # encoder2 (3 stages, DP=1, TP=2): all-stage groups
                    (4, 6, 8), (5, 7, 9),
                    # LLM (2 stages, DP=1, TP=4)
                    (10, 14), (11, 15), (12, 16), (13, 17),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(18)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1), (2, 3),
                    (4, 5), (6, 7), (8, 9),
                    (10, 11, 12, 13), (14, 15, 16, 17),
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder2 TP=4 SP=1, LLM TP=4 SP=1, DP=3
        # ------------------------------------------------------------------
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            {
                # encoder2: 3 stages, DP=3, TP=4  → 3×3 = 9 PP groups
                MultiModalProcessGroupMesh.pp_axis: [
                    tuple(range(i, i + 12 * 3, 12)) for i in range(12)
                ] + [
                    # LLM: 4 stages, DP=3, TP=4 → 3×4 = 12 PP groups
                    tuple(range(36 + i, 36 + i + 12 * 4, 12)) for i in range(12)
                ],
                MultiModalProcessGroupMesh.dp_axis: [
                    tuple([i, i + 4, i + 8])
                    for j in range(0, 84, 12)
                    for i in range(j, j + 4)
                ],
                MultiModalProcessGroupMesh.tp_axis: [
                    tuple(range(i, i + 4)) for i in range(0, 84, 4)
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder TP=2 SP=1, LLM TP=4 SP=2, DP=2
        # ------------------------------------------------------------------
        (
            40,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder: 2 stages, DP=2, SP=1, TP=2
                    (0, 4), (1, 5), (2, 6), (3, 7),
                    # LLM: 2 stages, DP=2, SP=2, TP=4
                    (8, 24), (9, 25), (10, 26), (11, 27),
                    (12, 28), (13, 29), (14, 30), (15, 31),
                    (16, 32), (17, 33), (18, 34), (19, 35),
                    (20, 36), (21, 37), (22, 38), (23, 39),
                ],
                MultiModalProcessGroupMesh.dp_axis: [
                    (0, 2), (1, 3), (4, 6), (5, 7),
                    (8, 16), (9, 17), (10, 18), (11, 19),
                    (12, 20), (13, 21), (14, 22), (15, 23),
                    (24, 32), (25, 33), (26, 34), (27, 35),
                    (28, 36), (29, 37), (30, 38), (31, 39),
                ],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1), (2, 3), (4, 5), (6, 7),
                    (8, 9, 10, 11), (12, 13, 14, 15),
                    (16, 17, 18, 19), (20, 21, 22, 23),
                    (24, 25, 26, 27), (28, 29, 30, 31),
                    (32, 33, 34, 35), (36, 37, 38, 39),
                ],
                MultiModalProcessGroupMesh.sp_axis: [(i,) for i in range(0, 8)]
                + [
                    (8, 12), (9, 13), (10, 14), (11, 15),
                    (16, 20), (17, 21), (18, 22), (19, 23),
                    (24, 28), (25, 29), (26, 30), (27, 31),
                    (32, 36), (33, 37), (34, 38), (35, 39),
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder1 TP=2 SP=1, encoder2 TP=2 SP=1, LLM TP=4 SP=4, DP=1
        # encoder2 has PP=3: all-stage PP groups
        # ------------------------------------------------------------------
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder1: 2 stages, DP=1, SP=1, TP=2
                    (0, 2), (1, 3),
                    # encoder2: 3 stages, DP=1, SP=1, TP=2 — full-pipeline groups
                    (4, 6, 8), (5, 7, 9),
                    # LLM: 2 stages, DP=1, SP=4, TP=4
                    (10, 26), (11, 27), (12, 28), (13, 29),
                    (14, 30), (15, 31), (16, 32), (17, 33),
                    (18, 34), (19, 35), (20, 36), (21, 37),
                    (22, 38), (23, 39), (24, 40), (25, 41),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(42)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1), (2, 3),
                    (4, 5), (6, 7), (8, 9),
                    (10, 11, 12, 13), (14, 15, 16, 17),
                    (18, 19, 20, 21), (22, 23, 24, 25),
                    (26, 27, 28, 29), (30, 31, 32, 33),
                    (34, 35, 36, 37), (38, 39, 40, 41),
                ],
                MultiModalProcessGroupMesh.sp_axis: [(i,) for i in range(0, 10)]
                + [
                    (10, 14, 18, 22), (11, 15, 19, 23),
                    (12, 16, 20, 24), (13, 17, 21, 25),
                    (26, 30, 34, 38), (27, 31, 35, 39),
                    (28, 32, 36, 40), (29, 33, 37, 41),
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder2 TP=4 SP=1, LLM TP=2 SP=4, DP=3
        # ------------------------------------------------------------------
        (
            132,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder2: 3 stages, DP=3, SP=1, TP=4
                    tuple(range(i, i + 12 * 3, 12)) for i in range(12)
                ] + [
                    # LLM: 4 stages, DP=3, SP=4, TP=2
                    # stride between stages = DP*SP*TP = 3*4*2 = 24
                    # 4 stages: first=36+i, last=36+i+72 → need 73 to include endpoint
                    tuple(range(36 + i, 36 + i + 73, 24)) for i in range(24)
                ],
                MultiModalProcessGroupMesh.tp_axis: [
                    (i, i + 1, i + 2, i + 3) for i in range(0, 36, 4)
                ] + [(i, i + 1) for i in range(36, 132, 2)],
                MultiModalProcessGroupMesh.dp_axis: [
                    (0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11),
                    (12, 16, 20), (13, 17, 21), (14, 18, 22), (15, 19, 23),
                    (24, 28, 32), (25, 29, 33), (26, 30, 34), (27, 31, 35),
                ] + [
                    (36 + i, 36 + i + 8, 36 + i + 16)
                    for j in range(0, 96, 24)
                    for i in range(j, j + 8)
                ],
                MultiModalProcessGroupMesh.sp_axis: [(i,) for i in range(0, 36)]
                + [
                    (n + i, n + i + 2, n + i + 4, n + i + 6)
                    for n in range(36, 132, 8)
                    for i in range(0, 2)
                ],
            },
        ),
        # ------------------------------------------------------------------
        # Case A (new): encoder TP=4 SP=1, LLM TP=2 SP=1, DP=1
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 4},
            (llm_template_2stages, 2, 1),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder: 2 stages, DP=1, SP=1, TP=4
                    (0, 4), (1, 5), (2, 6), (3, 7),
                    # LLM: 2 stages, DP=1, SP=1, TP=2
                    (8, 10), (9, 11),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(12)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1, 2, 3), (4, 5, 6, 7),
                    (8, 9), (10, 11),
                ],
            },
        ),
        # ------------------------------------------------------------------
        # Case B (new): encoder TP=2 SP=1, LLM TP=2 SP=2, DP=1
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 2),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder: 2 stages, DP=1, SP=1, TP=2
                    (0, 2), (1, 3),
                    # LLM: 2 stages, DP=1, SP=2, TP=2
                    (4, 8), (5, 9), (6, 10), (7, 11),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(12)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1), (2, 3),
                    (4, 5), (6, 7), (8, 9), (10, 11),
                ],
                MultiModalProcessGroupMesh.sp_axis: [
                    (0,), (1,), (2,), (3,),
                    (4, 6), (5, 7), (8, 10), (9, 11),
                ],
            },
        ),
        # ------------------------------------------------------------------
        # Case C (new): encoder TP=2 SP=2, LLM TP=2 SP=1, DP=1
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: (2, 2)},
            (llm_template_2stages, 2, 1),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder: 2 stages, DP=1, SP=2, TP=2
                    (0, 4), (1, 5), (2, 6), (3, 7),
                    # LLM: 2 stages, DP=1, SP=1, TP=2
                    (8, 10), (9, 11),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(12)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1), (2, 3), (4, 5), (6, 7),
                    (8, 9), (10, 11),
                ],
                MultiModalProcessGroupMesh.sp_axis: [
                    (0, 2), (1, 3), (4, 6), (5, 7),
                    (8,), (9,), (10,), (11,),
                ],
            },
        ),
        # ------------------------------------------------------------------
        # Case D (new): encoder TP=4 SP=2, LLM TP=2 SP=1, DP=1
        # ------------------------------------------------------------------
        (
            20,
            {encoder1_template: (4, 2)},
            (llm_template_2stages, 2, 1),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    # encoder: 2 stages, DP=1, SP=2, TP=4
                    (0, 8), (1, 9), (2, 10), (3, 11),
                    (4, 12), (5, 13), (6, 14), (7, 15),
                    # LLM: 2 stages, DP=1, SP=1, TP=2
                    (16, 18), (17, 19),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(20)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1, 2, 3), (4, 5, 6, 7),
                    (8, 9, 10, 11), (12, 13, 14, 15),
                    (16, 17), (18, 19),
                ],
                MultiModalProcessGroupMesh.sp_axis: [
                    (0, 4), (1, 5), (2, 6), (3, 7),
                    (8, 12), (9, 13), (10, 14), (11, 15),
                    (16,), (17,), (18,), (19,),
                ],
            },
        ),
        # ------------------------------------------------------------------
        # Minimal custom template (encoder PP=1 TP=2 SP=1, LLM PP=1 TP=2 SP=2, DP=1)
        # Both modals have PP=1 → PP groups are single-rank
        # ------------------------------------------------------------------
        (
            6,
            {PipelineTemplate("encoder1", [["layer.0", "layer.1"]]): 2},
            (PipelineTemplate("llm", [["layer.0", "layer.1"]]), 2, 2),
            {
                MultiModalProcessGroupMesh.pp_axis: [(0,), (1,), (2,), (3,), (4,), (5,)],
                MultiModalProcessGroupMesh.dp_axis: [
                    (0,), (1,), (2,), (3,), (4,), (5,),
                ],
                MultiModalProcessGroupMesh.tp_axis: [(0, 1), (2, 3), (4, 5)],
                MultiModalProcessGroupMesh.sp_axis: [(0,), (1,), (2, 4), (3, 5)],
            },
        ),
    ),
)
@pytest.mark.parametrize(
    "axis",
    [
        MultiModalProcessGroupMesh.pp_axis,
        MultiModalProcessGroupMesh.dp_axis,
        MultiModalProcessGroupMesh.tp_axis,
        MultiModalProcessGroupMesh.sp_axis,
    ],
    ids=["pp", "dp", "tp", "sp"],
)
def test_get_group_along_axis(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_group_ranks: dict[int, list[tuple[int, ...]]],
    axis: int,
):
    if axis not in expected_group_ranks:
        pytest.skip("Axis not in expected_group_ranks")

    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
        mesh.get_group_along_axis(axis)

        assert list(mesh._ranks_to_group.keys()) == expected_group_ranks[axis], (
            f"rank {rank}: got {list(mesh._ranks_to_group.keys())}"
        )
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template",
    [
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
        ),
        (
            16,
            {encoder1_template: (2, 2)},
            (llm_template_2stages, 2, 2),
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
        ),
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
        ),
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
        ),
        (
            132,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
        ),
        (
            6,
            {PipelineTemplate("encoder1", [["layer.0", "layer.1"]]): 2},
            (PipelineTemplate("llm", [["layer.0", "layer.1"]]), 2, 2),
        ),
        # New cases A–D
        (12, {encoder1_template: 4}, (llm_template_2stages, 2, 1)),
        (12, {encoder1_template: 2}, (llm_template_2stages, 2, 2)),
        (12, {encoder1_template: (2, 2)}, (llm_template_2stages, 2, 1)),
        (20, {encoder1_template: (4, 2)}, (llm_template_2stages, 2, 1)),
    ],
)
def test_create_group_along_axis_order(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    mocker: MockerFixture,
):
    recorded_new_group_calls: dict[int, list] = defaultdict(list)

    def record_new_group_call_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # dist.new_group is called with ranks= as a keyword argument
            ranks = args[0] if args else kwargs.get("ranks", [])
            recorded_new_group_calls[dist.get_rank()].append(tuple(sorted(ranks)))
            return func(*args, **kwargs)

        return wrapper

    mocker.patch.object(
        dist,
        "new_group",
        wraps=record_new_group_call_decorator(dist.new_group),
    )

    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
        mesh.get_group_along_axis(mesh.pp_axis)
        mesh.get_group_along_axis(mesh.dp_axis)
        mesh.get_group_along_axis(mesh.tp_axis)
        mesh.get_group_along_axis(mesh.sp_axis)

        dist.destroy_process_group()

    for rank, calls in recorded_new_group_calls.items():
        assert calls == recorded_new_group_calls[0]
