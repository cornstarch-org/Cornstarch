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


@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_mesh, expected_ranks",
    [
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            [
                [
                    [[0, 0, 1, 1]],
                    [[2, 2, 3, 3]],
                ],
                [
                    [[4, 4, 5, 5]],
                    [[6, 6, 7, 7]],
                ],
                [
                    [[8, 9, 10, 11]],
                    [[12, 13, 14, 15]],
                ],
                [
                    [[16, 17, 18, 19]],
                    [[20, 21, 22, 23]],
                ],
            ],
            {
                encoder1_template: list(range(0, 8)),
                llm_template_2stages: list(range(8, 24)),
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            [
                [
                    [[0, 0, 1, 1]],
                ],
                [
                    [[2, 2, 3, 3]],
                ],
                [
                    [[4, 4, 5, 5]],
                ],
                [
                    [[6, 6, 7, 7]],
                ],
                [
                    [[8, 8, 9, 9]],
                ],
                [
                    [[10, 11, 12, 13]],
                ],
                [
                    [[14, 15, 16, 17]],
                ],
            ],
            {
                encoder1_template: list(range(0, 4)),
                encoder2_template: list(range(4, 10)),
                llm_template_2stages: list(range(10, 18)),
            },
        ),
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            [
                [
                    [[0, 1, 2, 3]],
                    [[4, 5, 6, 7]],
                    [[8, 9, 10, 11]],
                ],
                [
                    [[12, 13, 14, 15]],
                    [[16, 17, 18, 19]],
                    [[20, 21, 22, 23]],
                ],
                [
                    [[24, 25, 26, 27]],
                    [[28, 29, 30, 31]],
                    [[32, 33, 34, 35]],
                ],
                [
                    [[36, 37, 38, 39]],
                    [[40, 41, 42, 43]],
                    [[44, 45, 46, 47]],
                ],
                [
                    [[48, 49, 50, 51]],
                    [[52, 53, 54, 55]],
                    [[56, 57, 58, 59]],
                ],
                [
                    [[60, 61, 62, 63]],
                    [[64, 65, 66, 67]],
                    [[68, 69, 70, 71]],
                ],
                [
                    [[72, 73, 74, 75]],
                    [[76, 77, 78, 79]],
                    [[80, 81, 82, 83]],
                ],
            ],
            {
                encoder2_template: list(range(0, 36)),
                llm_template_4stages: list(range(36, 84)),
            },
        ),
        (
            40,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            [
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    [[2, 2, 3, 3], [2, 2, 3, 3]],
                ],
                [
                    [[4, 4, 5, 5], [4, 4, 5, 5]],
                    [[6, 6, 7, 7], [6, 6, 7, 7]],
                ],
                [
                    [[8, 9, 10, 11], [12, 13, 14, 15]],
                    [[16, 17, 18, 19], [20, 21, 22, 23]],
                ],
                [
                    [[24, 25, 26, 27], [28, 29, 30, 31]],
                    [[32, 33, 34, 35], [36, 37, 38, 39]],
                ],
            ],
            {
                encoder1_template: list(range(0, 8)),
                llm_template_2stages: list(range(8, 40)),
            },
        ),
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            [
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                ],
                [
                    [[2, 2, 3, 3], [2, 2, 3, 3], [2, 2, 3, 3], [2, 2, 3, 3]],
                ],
                [
                    [[4, 4, 5, 5], [4, 4, 5, 5], [4, 4, 5, 5], [4, 4, 5, 5]],
                ],
                [
                    [[6, 6, 7, 7], [6, 6, 7, 7], [6, 6, 7, 7], [6, 6, 7, 7]],
                ],
                [
                    [[8, 8, 9, 9], [8, 8, 9, 9], [8, 8, 9, 9], [8, 8, 9, 9]],
                ],
                [
                    [
                        [10, 11, 12, 13],
                        [14, 15, 16, 17],
                        [18, 19, 20, 21],
                        [22, 23, 24, 25],
                    ],
                ],
                [
                    [
                        [26, 27, 28, 29],
                        [30, 31, 32, 33],
                        [34, 35, 36, 37],
                        [38, 39, 40, 41],
                    ],
                ],
            ],
            {
                encoder1_template: list(range(0, 4)),
                encoder2_template: list(range(4, 10)),
                llm_template_2stages: list(range(10, 42)),
            },
        ),
        (
            132,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            [
                [
                    [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                    [[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]],
                    [[8, 9, 10, 11], [8, 9, 10, 11], [8, 9, 10, 11], [8, 9, 10, 11]],
                ],
                [
                    [
                        [12, 13, 14, 15],
                        [12, 13, 14, 15],
                        [12, 13, 14, 15],
                        [12, 13, 14, 15],
                    ],
                    [
                        [16, 17, 18, 19],
                        [16, 17, 18, 19],
                        [16, 17, 18, 19],
                        [16, 17, 18, 19],
                    ],
                    [
                        [20, 21, 22, 23],
                        [20, 21, 22, 23],
                        [20, 21, 22, 23],
                        [20, 21, 22, 23],
                    ],
                ],
                [
                    [
                        [24, 25, 26, 27],
                        [24, 25, 26, 27],
                        [24, 25, 26, 27],
                        [24, 25, 26, 27],
                    ],
                    [
                        [28, 29, 30, 31],
                        [28, 29, 30, 31],
                        [28, 29, 30, 31],
                        [28, 29, 30, 31],
                    ],
                    [
                        [32, 33, 34, 35],
                        [32, 33, 34, 35],
                        [32, 33, 34, 35],
                        [32, 33, 34, 35],
                    ],
                ],
                [
                    [
                        [36, 36, 37, 37],
                        [38, 38, 39, 39],
                        [40, 40, 41, 41],
                        [42, 42, 43, 43],
                    ],
                    [
                        [44, 44, 45, 45],
                        [46, 46, 47, 47],
                        [48, 48, 49, 49],
                        [50, 50, 51, 51],
                    ],
                    [
                        [52, 52, 53, 53],
                        [54, 54, 55, 55],
                        [56, 56, 57, 57],
                        [58, 58, 59, 59],
                    ],
                ],
                [
                    [
                        [60, 60, 61, 61],
                        [62, 62, 63, 63],
                        [64, 64, 65, 65],
                        [66, 66, 67, 67],
                    ],
                    [
                        [68, 68, 69, 69],
                        [70, 70, 71, 71],
                        [72, 72, 73, 73],
                        [74, 74, 75, 75],
                    ],
                    [
                        [76, 76, 77, 77],
                        [78, 78, 79, 79],
                        [80, 80, 81, 81],
                        [82, 82, 83, 83],
                    ],
                ],
                [
                    [
                        [84, 84, 85, 85],
                        [86, 86, 87, 87],
                        [88, 88, 89, 89],
                        [90, 90, 91, 91],
                    ],
                    [
                        [92, 92, 93, 93],
                        [94, 94, 95, 95],
                        [96, 96, 97, 97],
                        [98, 98, 99, 99],
                    ],
                    [
                        [100, 100, 101, 101],
                        [102, 102, 103, 103],
                        [104, 104, 105, 105],
                        [106, 106, 107, 107],
                    ],
                ],
                [
                    [
                        [108, 108, 109, 109],
                        [110, 110, 111, 111],
                        [112, 112, 113, 113],
                        [114, 114, 115, 115],
                    ],
                    [
                        [116, 116, 117, 117],
                        [118, 118, 119, 119],
                        [120, 120, 121, 121],
                        [122, 122, 123, 123],
                    ],
                    [
                        [124, 124, 125, 125],
                        [126, 126, 127, 127],
                        [128, 128, 129, 129],
                        [130, 130, 131, 131],
                    ],
                ],
            ],
            {
                encoder2_template: list(range(0, 36)),
                llm_template_4stages: list(range(36, 132)),
            },
        ),
    ],
)
def test_init_process_group_mesh(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_mesh: list[list[list[list[int]]]],
    expected_ranks: dict[PipelineTemplate, list[int]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
        assert (mesh.mesh == expected_mesh).all()
        assert mesh.modal_to_ranks == expected_ranks

        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_group_ranks",
    (
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    (0, 4, 8, 16),
                    (0, 4, 9, 17),
                    (1, 5, 10, 18),
                    (1, 5, 11, 19),
                    (2, 6, 12, 20),
                    (2, 6, 13, 21),
                    (3, 7, 14, 22),
                    (3, 7, 15, 23),
                ],
                MultiModalProcessGroupMesh.dp_axis: [
                    (0, 2),
                    (1, 3),
                    (4, 6),
                    (5, 7),
                    (8, 12),
                    (9, 13),
                    (10, 14),
                    (11, 15),
                    (16, 20),
                    (17, 21),
                    (18, 22),
                    (19, 23),
                ],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9, 10, 11),
                    (12, 13, 14, 15),
                    (16, 17, 18, 19),
                    (20, 21, 22, 23),
                ],
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    (0, 2, 4, 6, 8, 10, 14),
                    (0, 2, 4, 6, 8, 11, 15),
                    (1, 3, 5, 7, 9, 12, 16),
                    (1, 3, 5, 7, 9, 13, 17),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(18)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9),
                    (10, 11, 12, 13),
                    (14, 15, 16, 17),
                ],
            },
        ),
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            {
                # (0, 12, 24, 36, 48, 60, 72), (1, 13, 25, 37, 49, 61, 73), ...
                MultiModalProcessGroupMesh.pp_axis: [
                    tuple(range(i, i + 12 * 7, 12)) for i in range(12)
                ],
                # (0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11), (12, 16, 20), (13, 17, 21), ...
                MultiModalProcessGroupMesh.dp_axis: [
                    tuple([i, i + 4, i + 8])
                    for j in range(0, 84, 12)
                    for i in range(j, j + 4)
                ],
                # (0, 1, 2, 3), (4, 5, 6, 7), ...
                MultiModalProcessGroupMesh.tp_axis: [
                    tuple(range(i, i + 4)) for i in range(0, 84, 4)
                ],
            },
        ),
        (
            40,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    (0, 4, 8, 24),
                    (0, 4, 9, 25),
                    (1, 5, 10, 26),
                    (1, 5, 11, 27),
                    (0, 4, 12, 28),
                    (0, 4, 13, 29),
                    (1, 5, 14, 30),
                    (1, 5, 15, 31),
                    (2, 6, 16, 32),
                    (2, 6, 17, 33),
                    (3, 7, 18, 34),
                    (3, 7, 19, 35),
                    (2, 6, 20, 36),
                    (2, 6, 21, 37),
                    (3, 7, 22, 38),
                    (3, 7, 23, 39),
                ],
                MultiModalProcessGroupMesh.dp_axis: [
                    (0, 2),
                    (1, 3),
                    (4, 6),
                    (5, 7),
                    (8, 16),
                    (9, 17),
                    (10, 18),
                    (11, 19),
                    (12, 20),
                    (13, 21),
                    (14, 22),
                    (15, 23),
                    (24, 32),
                    (25, 33),
                    (26, 34),
                    (27, 35),
                    (28, 36),
                    (29, 37),
                    (30, 38),
                    (31, 39),
                ],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9, 10, 11),
                    (12, 13, 14, 15),
                    (16, 17, 18, 19),
                    (20, 21, 22, 23),
                    (24, 25, 26, 27),
                    (28, 29, 30, 31),
                    (32, 33, 34, 35),
                    (36, 37, 38, 39),
                ],
                MultiModalProcessGroupMesh.sp_axis: [(i,) for i in range(0, 8)]
                + [
                    (8, 12),
                    (9, 13),
                    (10, 14),
                    (11, 15),
                    (16, 20),
                    (17, 21),
                    (18, 22),
                    (19, 23),
                    (24, 28),
                    (25, 29),
                    (26, 30),
                    (27, 31),
                    (32, 36),
                    (33, 37),
                    (34, 38),
                    (35, 39),
                ],
            },
        ),
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    (0, 2, 4, 6, 8, 10, 26),
                    (0, 2, 4, 6, 8, 11, 27),
                    (1, 3, 5, 7, 9, 12, 28),
                    (1, 3, 5, 7, 9, 13, 29),
                    (0, 2, 4, 6, 8, 14, 30),
                    (0, 2, 4, 6, 8, 15, 31),
                    (1, 3, 5, 7, 9, 16, 32),
                    (1, 3, 5, 7, 9, 17, 33),
                    (0, 2, 4, 6, 8, 18, 34),
                    (0, 2, 4, 6, 8, 19, 35),
                    (1, 3, 5, 7, 9, 20, 36),
                    (1, 3, 5, 7, 9, 21, 37),
                    (0, 2, 4, 6, 8, 22, 38),
                    (0, 2, 4, 6, 8, 23, 39),
                    (1, 3, 5, 7, 9, 24, 40),
                    (1, 3, 5, 7, 9, 25, 41),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(42)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9),
                    (10, 11, 12, 13),
                    (14, 15, 16, 17),
                    (18, 19, 20, 21),
                    (22, 23, 24, 25),
                    (26, 27, 28, 29),
                    (30, 31, 32, 33),
                    (34, 35, 36, 37),
                    (38, 39, 40, 41),
                ],
                MultiModalProcessGroupMesh.sp_axis: [(i,) for i in range(0, 10)]
                + [
                    (10, 14, 18, 22),
                    (11, 15, 19, 23),
                    (12, 16, 20, 24),
                    (13, 17, 21, 25),
                    (26, 30, 34, 38),
                    (27, 31, 35, 39),
                    (28, 32, 36, 40),
                    (29, 33, 37, 41),
                ],
            },
        ),
        (
            132,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    (0, 12, 24, 36, 60, 84, 108),
                    (1, 13, 25, 36, 60, 84, 108),
                    (2, 14, 26, 37, 61, 85, 109),
                    (3, 15, 27, 37, 61, 85, 109),
                    (0, 12, 24, 38, 62, 86, 110),
                    (1, 13, 25, 38, 62, 86, 110),
                    (2, 14, 26, 39, 63, 87, 111),
                    (3, 15, 27, 39, 63, 87, 111),
                    (0, 12, 24, 40, 64, 88, 112),
                    (1, 13, 25, 40, 64, 88, 112),
                    (2, 14, 26, 41, 65, 89, 113),
                    (3, 15, 27, 41, 65, 89, 113),
                    (0, 12, 24, 42, 66, 90, 114),
                    (1, 13, 25, 42, 66, 90, 114),
                    (2, 14, 26, 43, 67, 91, 115),
                    (3, 15, 27, 43, 67, 91, 115),
                    (4, 16, 28, 44, 68, 92, 116),
                    (5, 17, 29, 44, 68, 92, 116),
                    (6, 18, 30, 45, 69, 93, 117),
                    (7, 19, 31, 45, 69, 93, 117),
                    (4, 16, 28, 46, 70, 94, 118),
                    (5, 17, 29, 46, 70, 94, 118),
                    (6, 18, 30, 47, 71, 95, 119),
                    (7, 19, 31, 47, 71, 95, 119),
                    (4, 16, 28, 48, 72, 96, 120),
                    (5, 17, 29, 48, 72, 96, 120),
                    (6, 18, 30, 49, 73, 97, 121),
                    (7, 19, 31, 49, 73, 97, 121),
                    (4, 16, 28, 50, 74, 98, 122),
                    (5, 17, 29, 50, 74, 98, 122),
                    (6, 18, 30, 51, 75, 99, 123),
                    (7, 19, 31, 51, 75, 99, 123),
                    (8, 20, 32, 52, 76, 100, 124),
                    (9, 21, 33, 52, 76, 100, 124),
                    (10, 22, 34, 53, 77, 101, 125),
                    (11, 23, 35, 53, 77, 101, 125),
                    (8, 20, 32, 54, 78, 102, 126),
                    (9, 21, 33, 54, 78, 102, 126),
                    (10, 22, 34, 55, 79, 103, 127),
                    (11, 23, 35, 55, 79, 103, 127),
                    (8, 20, 32, 56, 80, 104, 128),
                    (9, 21, 33, 56, 80, 104, 128),
                    (10, 22, 34, 57, 81, 105, 129),
                    (11, 23, 35, 57, 81, 105, 129),
                    (8, 20, 32, 58, 82, 106, 130),
                    (9, 21, 33, 58, 82, 106, 130),
                    (10, 22, 34, 59, 83, 107, 131),
                    (11, 23, 35, 59, 83, 107, 131),
                ],
                MultiModalProcessGroupMesh.tp_axis: [
                    (i, i + 1, i + 2, i + 3) for i in range(0, 36, 4)
                ]
                + [(i, i + 1) for i in range(36, 132, 2)],
                MultiModalProcessGroupMesh.dp_axis: [
                    (0, 4, 8),
                    (1, 5, 9),
                    (2, 6, 10),
                    (3, 7, 11),
                    (12, 16, 20),
                    (13, 17, 21),
                    (14, 18, 22),
                    (15, 19, 23),
                    (24, 28, 32),
                    (25, 29, 33),
                    (26, 30, 34),
                    (27, 31, 35),
                    (36, 44, 52),
                    (37, 45, 53),
                    (38, 46, 54),
                    (39, 47, 55),
                    (40, 48, 56),
                    (41, 49, 57),
                    (42, 50, 58),
                    (43, 51, 59),
                    (60, 68, 76),
                    (61, 69, 77),
                    (62, 70, 78),
                    (63, 71, 79),
                    (64, 72, 80),
                    (65, 73, 81),
                    (66, 74, 82),
                    (67, 75, 83),
                    (84, 92, 100),
                    (85, 93, 101),
                    (86, 94, 102),
                    (87, 95, 103),
                    (88, 96, 104),
                    (89, 97, 105),
                    (90, 98, 106),
                    (91, 99, 107),
                    (108, 116, 124),
                    (109, 117, 125),
                    (110, 118, 126),
                    (111, 119, 127),
                    (112, 120, 128),
                    (113, 121, 129),
                    (114, 122, 130),
                    (115, 123, 131),
                ],
                MultiModalProcessGroupMesh.sp_axis: [(i,) for i in range(0, 36)]
                + [
                    (n + i, n + i + 2, n + i + 4, n + i + 6)
                    for n in range(36, 132, 8)
                    for i in range(0, 2)
                ],
            },
        ),
        (
            6,
            {PipelineTemplate("encoder1", [["layer.0", "layer.1"]]): 2},
            (PipelineTemplate("llm", [["layer.0", "layer.1"]]), 2, 2),
            {
                MultiModalProcessGroupMesh.pp_axis: [(0, 2), (1, 3), (0, 4), (1, 5)],
                MultiModalProcessGroupMesh.dp_axis: [
                    (0,),
                    (1,),
                    (2,),
                    (3,),
                    (4,),
                    (5,),
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
    mocker: MockerFixture,
):
    if axis not in expected_group_ranks:
        pytest.skip("Axis not in expected_group_ranks")

    recorded_new_group_calls: dict[int, list] = defaultdict(list)

    def record_new_group_call_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recorded_new_group_calls[dist.get_rank()].append(args[0])
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
        mesh.get_group_along_axis(axis)

        assert list(mesh._ranks_to_group.keys()) == expected_group_ranks[axis]
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
            recorded_new_group_calls[dist.get_rank()].append(args[0])
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
