from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from ..gemma import GemmaForCausalLMBase, GemmaModelBase
from ._utils import ColossalaiHybridParallelBase


@instantiate_parametrized_tests
class TestGemmaModelPolicyClass(ColossalaiHybridParallelBase):
    @parametrize("tp_size, pp_size", [(4, 1), (1, 1), (2, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.set_model(GemmaModelBase())
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)

    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    def test_context_parallel(self, tp_size: int, pp_size: int):
        # ring_attn is for causal lm only
        self.set_model(GemmaModelBase())
        self.run_hybrid_parallel(tp_size, pp_size, "all_to_all", True, "bf16")


@instantiate_parametrized_tests
class TestGemmaForCausalLMPolicyClass(ColossalaiHybridParallelBase):
    @parametrize("tp_size, pp_size", [(4, 1), (1, 1), (2, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.set_model(GemmaForCausalLMBase())
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)

    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    def test_context_parallel(self, tp_size: int, pp_size: int):
        # ring_attn is for causal lm only
        self.set_model(GemmaForCausalLMBase())
        self.run_hybrid_parallel(tp_size, pp_size, "all_to_all", True, "bf16")
