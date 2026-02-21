import os
import re
import unittest
from unittest.mock import patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from .model_zoo import (
    CLIPModelBase,
    LlamaForCausalLMBase,
    Qwen2ForCausalLMBase,
    Qwen2VisionTransformerBase,
)
from .utils import CornstarchMultimodalParallelBase

vision_models = dict(
    clip=CLIPModelBase,
    qwen2_vision=Qwen2VisionTransformerBase,
)

causal_lms = dict(
    llama=LlamaForCausalLMBase,
    qwen2=Qwen2ForCausalLMBase,
)


@instantiate_parametrized_tests
class VisionLanguagePipeweaverParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        tp = int(os.environ["TP"])
        pp = int(os.environ["PP"])
        sp = int(os.environ["SP"])
        return tp * pp * sp

    def setUp(self) -> None:
        tp = re.search(r"tp=(\d+)", self._testMethodName)
        pp = re.search(r"pp=(\d+)", self._testMethodName)
        sp = re.search(r"sp=(\d+)", self._testMethodName)
        assert all(
            [tp, pp, sp]
        ), f"Could not parse parallelism params from {self._testMethodName}"

        tp = tp.group(1)
        pp = pp.group(1)
        sp = sp.group(1)

        assert int(pp) > 1, "Pipeweaver requires at least 2 stages of PP"
        with patch.dict(
            os.environ,
            {"TP": tp, "PP": pp, "SP": sp},
        ):
            super().setUp()

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("tp", [1, 2], name_fn=lambda x: f"tp={x}")
    @parametrize("pp", [2, 4], name_fn=lambda x: f"pp={x}")
    @parametrize("sp", [1, 2], name_fn=lambda x: f"sp={x}")
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        tp: int,
        pp: int,
        sp: int,
    ) -> None:
        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_pipeweaver_parallel(tp_size=tp, pp_size=pp, sp_size=sp)
