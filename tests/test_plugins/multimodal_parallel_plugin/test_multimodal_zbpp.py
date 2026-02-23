import os
import re
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.plugin.multimodal_parallel_plugin import (
    MultiModalPipelineStageManager,
    MultiModalProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_zbpp import (
    MultimodalEncoderTrainingZeroBubblePipelineSchedule,
)

from ...distributed_base import GlooDistributedTestBase
from ..common import encoder1_template, llm_template_2stages


def _expected_event_sequence(i: int, p: int, num_microbatches: int) -> list[str]:
    warmup1 = (p - i - 1) * 2
    warmup2_end = (p - 1) * 2
    warmup2_iters = warmup2_end - warmup1
    steady_iters = num_microbatches - warmup2_end
    cooldown1_iters = num_microbatches - (warmup2_iters + steady_iters)
    cooldown2_iters = num_microbatches - (steady_iters + cooldown1_iters)

    return (
        ["F"] * warmup1
        + ["F", "BI"] * warmup2_iters
        + ["F", "BI", "BP"] * steady_iters
        + ["BI", "BP"] * cooldown1_iters
        + ["BP"] * cooldown2_iters
    )


@instantiate_parametrized_tests
class TestMultimodalZBPPBasic(GlooDistributedTestBase):
    _CONFIGS = [
        (1, 1, 1, 1),
        (2, 1, 4, 1),
        (2, 2, 2, 1),
        (2, 1, 2, 2),
        (2, 1, 4, 2),
        (2, 2, 4, 2),
        (2, 2, 2, 4),
        (2, 4, 1, 2),
    ]

    @property
    def world_size(self):
        enc_tp = int(os.environ["ENC_TP"])
        enc_sp = int(os.environ["ENC_SP"])
        llm_tp = int(os.environ["LLM_TP"])
        llm_sp = int(os.environ["LLM_SP"])
        dp = 1
        enc_ranks = encoder1_template.num_stages * enc_tp * enc_sp * dp
        llm_ranks = llm_template_2stages.num_stages * llm_tp * llm_sp * dp
        return enc_ranks + llm_ranks

    def setUp(self) -> None:
        pattern = r"enc_tp=(\d+)_enc_sp=(\d+)_llm_tp=(\d+)_llm_sp=(\d+)"
        match = re.search(pattern, self._testMethodName)
        assert (
            match is not None
        ), f"Could not parse parallelism params from {self._testMethodName}"
        with patch.dict(
            os.environ,
            {
                "ENC_TP": match.group(1),
                "ENC_SP": match.group(2),
                "LLM_TP": match.group(3),
                "LLM_SP": match.group(4),
            },
        ):
            super().setUp()

    def _create_schedule(
        self,
        num_microbatches: int,
        enc_tp: int,
        enc_sp: int,
        llm_tp: int,
        llm_sp: int,
    ) -> MultimodalEncoderTrainingZeroBubblePipelineSchedule:
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={encoder1_template: (enc_tp, enc_sp)},
            llm_template=(llm_template_2stages, llm_tp, llm_sp),
        )
        stage_manager = MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)
        schedule = MultimodalEncoderTrainingZeroBubblePipelineSchedule(
            stage_manager=stage_manager,
            num_microbatches=num_microbatches,
            microbatch_size=1,
            encoder_sp_gather=False,
            encoder_tp_hidden_scatter=enc_tp > llm_tp,
        )
        schedule.forward_only = False
        return schedule

    @parametrize(
        "enc_tp,enc_sp,llm_tp,llm_sp",
        _CONFIGS,
        name_fn=lambda enc_tp, enc_sp, llm_tp, llm_sp: (
            f"enc_tp={enc_tp}_enc_sp={enc_sp}_llm_tp={llm_tp}_llm_sp={llm_sp}"
        ),
    )
    def test_global_stage_info(
        self, enc_tp: int, enc_sp: int, llm_tp: int, llm_sp: int
    ):
        """Validate global stage indexing across heterogeneous TP/SP layouts.

        Ensures all ranks agree on total pipeline depth `p=4`, cover stage ids
        {0,1,2,3}, and have expected per-stage rank multiplicities from TP/SP.
        """
        schedule = self._create_schedule(
            num_microbatches=8,
            enc_tp=enc_tp,
            enc_sp=enc_sp,
            llm_tp=llm_tp,
            llm_sp=llm_sp,
        )
        i, p = schedule._get_global_stage_info()

        payload = torch.tensor([i, p], dtype=torch.int64)
        gathered = [torch.zeros_like(payload) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, payload, group=dist.group.WORLD)

        assert p == 4
        gathered_i = [int(t[0].item()) for t in gathered]
        assert sorted(set(gathered_i)) == [0, 1, 2, 3]
        assert gathered_i.count(0) == enc_tp * enc_sp
        assert gathered_i.count(1) == enc_tp * enc_sp
        assert gathered_i.count(2) == llm_tp * llm_sp
        assert gathered_i.count(3) == llm_tp * llm_sp
        assert all(int(t[1].item()) == 4 for t in gathered)

    @parametrize(
        "enc_tp,enc_sp,llm_tp,llm_sp",
        _CONFIGS,
        name_fn=lambda enc_tp, enc_sp, llm_tp, llm_sp: (
            f"enc_tp={enc_tp}_enc_sp={enc_sp}_llm_tp={llm_tp}_llm_sp={llm_sp}"
        ),
    )
    def test_num_microbatches_lower_bound(
        self, enc_tp: int, enc_sp: int, llm_tp: int, llm_sp: int
    ):
        """Enforce ZBPP precondition: num_microbatches must satisfy N >= 2*p."""
        schedule = self._create_schedule(
            num_microbatches=7,  # For p=4, ZBPP requires at least 2*p=8.
            enc_tp=enc_tp,
            enc_sp=enc_sp,
            llm_tp=llm_tp,
            llm_sp=llm_sp,
        )
        with pytest.raises(ValueError, match="requires num_microbatches >="):
            schedule.run_forward_backward(
                model=torch.nn.Identity(),
                data_iter=iter([{}]),
                criterion=lambda _output, _micro_batch: torch.tensor(0.0),
                optimizer=object(),
                return_loss=False,
                return_outputs=False,
            )

    @parametrize(
        "enc_tp,enc_sp,llm_tp,llm_sp",
        _CONFIGS,
        name_fn=lambda enc_tp, enc_sp, llm_tp, llm_sp: (
            f"enc_tp={enc_tp}_enc_sp={enc_sp}_llm_tp={llm_tp}_llm_sp={llm_sp}"
        ),
    )
    def test_phase_event_order(
        self, enc_tp: int, enc_sp: int, llm_tp: int, llm_sp: int
    ):
        """Check phase-level execution order matches ZBPP schedule formula.

        The test patches runtime comm/compute hooks and records symbolic events:
        F (forward), BI (input-grad backward), BP (weight-grad backward).
        """
        num_microbatches = 8
        schedule = self._create_schedule(
            num_microbatches=num_microbatches,
            enc_tp=enc_tp,
            enc_sp=enc_sp,
            llm_tp=llm_tp,
            llm_sp=llm_sp,
        )
        i, p = schedule._get_global_stage_info()

        events: list[str] = []

        # Patch runtime methods to validate scheduling order in isolation.
        schedule.load_batch = lambda *_args, **_kwargs: None
        schedule.recv_forward = lambda: {"x": torch.tensor([1.0])}
        schedule.send_forward = lambda _output_obj: None
        schedule.recv_backward = lambda: {"x": torch.tensor([1.0])}
        schedule.send_backward = lambda _input_obj, _input_obj_grad: None
        schedule.forward_step = (
            lambda _model, _input_obj, _criterion, _accum_loss=None, _outputs=None: (
                events.append("F"),
                {"y": torch.tensor([1.0])},
            )[1]
        )
        schedule.backward_b_step = (
            lambda _model, _optimizer, _input_obj, _output_obj, _output_obj_grad: (
                events.append("BI"),
                {"x": torch.tensor([1.0])},
            )[1]
        )
        schedule.backward_w_step = lambda: events.append("BP")

        result = schedule.run_forward_backward(
            model=torch.nn.Identity(),
            data_iter=iter([{}]),
            criterion=lambda _output, _micro_batch: torch.tensor(0.0),
            optimizer=object(),
            return_loss=False,
            return_outputs=False,
        )

        assert result["loss"] is None
        assert result["outputs"] is None
        assert events == _expected_event_sequence(i, p, num_microbatches)
