from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from ..intern_vit import InternVisonModelBase
from ._utils import ColossalaiHybridParallelBase


@instantiate_parametrized_tests
class TestInternVisionModelPolicyClass(ColossalaiHybridParallelBase):
    @parametrize("tp_size, pp_size", [(4, 1), (1, 1), (2, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.set_model(InternVisonModelBase())
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)
