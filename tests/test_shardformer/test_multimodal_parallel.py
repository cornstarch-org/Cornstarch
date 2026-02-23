import os
import re
from unittest.mock import patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from .model_zoo import (
    CLIPModelBase,
    LlamaForCausalLMBase,
    MistralForCausalLMBase,
    Phi3ForCausalLMBase,
    Phi4MultimodalAudioModelBase,
    Qwen2AudioEncoderBase,
    Qwen2ForCausalLMBase,
    Qwen2VisionTransformerBase,
    SiglipModelBase,
    WhisperEncoderBase,
)
from .utils import CornstarchMultimodalParallelBase

vision_models = dict(
    clip=CLIPModelBase,
    siglip=SiglipModelBase,
    qwen2_vision=Qwen2VisionTransformerBase,
    # evaclip=EvaCLIPModelBase,
)

audio_models = dict(
    qwen2_audio=Qwen2AudioEncoderBase,
    whisper=WhisperEncoderBase,
    phi4_audio=Phi4MultimodalAudioModelBase,
)

causal_lms = dict(
    # gemma2=Gemma2ForCausalLMBase,
    llama=LlamaForCausalLMBase,
    # mistral=MistralForCausalLMBase,
    # phi3=Phi3ForCausalLMBase,
    qwen2=Qwen2ForCausalLMBase,
    # gemma=GemmaForCausalLMBase,
    # internlm2=InternLM2ForCausalLMBase,
    # mixtral=MixtralForCausalLMBase,
)


@instantiate_parametrized_tests
class VisionLanguageMultimodalParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        vtp = int(os.environ["VTP"])
        vpp = int(os.environ["VPP"])
        ltp = int(os.environ["LTP"])
        lpp = int(os.environ["LPP"])
        return vpp * vtp + lpp * ltp

    def setUp(self) -> None:
        vtp = re.search(r"vtp=(\d+)", self._testMethodName)
        vpp = re.search(r"vpp=(\d+)", self._testMethodName)
        ltp = re.search(r"ltp=(\d+)", self._testMethodName)
        lpp = re.search(r"lpp=(\d+)", self._testMethodName)
        assert all(
            [vtp, vpp, ltp, lpp]
        ), f"Could not parse parallelism params from {self._testMethodName}"
        with patch.dict(
            os.environ,
            {
                "VTP": vtp.group(1),
                "VPP": vpp.group(1),
                "LTP": ltp.group(1),
                "LPP": lpp.group(1),
            },
        ):
            super().setUp()

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("vtp", [1, 2], name_fn=lambda x: f"vtp={x}")
    @parametrize("vpp", [1, 2], name_fn=lambda x: f"vpp={x}")
    @parametrize("ltp", [1, 2], name_fn=lambda x: f"ltp={x}")
    @parametrize("lpp", [1, 2], name_fn=lambda x: f"lpp={x}")
    @parametrize(
        "pipeline_schedule",
        ["1f1b", "zbpp"],
        name_fn=lambda x: f"pp={x}",
    )
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        vtp: int,
        vpp: int,
        ltp: int,
        lpp: int,
        pipeline_schedule: str,
    ):
        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            {"vision": vtp, "llm": ltp},
            {"vision": vpp, "llm": lpp},
            pipeline_schedule=pipeline_schedule,
        )


@instantiate_parametrized_tests
class VisionLanguageMultimodalContextParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        vtp = int(os.environ["VTP"])
        vpp = int(os.environ["VPP"])
        vsp = int(os.environ["VSP"])
        ltp = int(os.environ["LTP"])
        lpp = int(os.environ["LPP"])
        lsp = int(os.environ["LSP"])
        return vpp * vtp * vsp + lpp * ltp * lsp

    def setUp(self) -> None:
        vtp = re.search(r"vtp=(\d+)", self._testMethodName)
        vpp = re.search(r"vpp=(\d+)", self._testMethodName)
        vsp = re.search(r"vsp=(\d+)", self._testMethodName)
        ltp = re.search(r"ltp=(\d+)", self._testMethodName)
        lpp = re.search(r"lpp=(\d+)", self._testMethodName)
        lsp = re.search(r"lsp=(\d+)", self._testMethodName)
        assert all(
            [vtp, vpp, vsp, ltp, lpp, lsp]
        ), f"Could not parse parallelism params from {self._testMethodName}"
        with patch.dict(
            os.environ,
            {
                "VTP": vtp.group(1),
                "VPP": vpp.group(1),
                "VSP": vsp.group(1),
                "LTP": ltp.group(1),
                "LPP": lpp.group(1),
                "LSP": lsp.group(1),
            },
        ):
            super().setUp()

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("vtp", [1, 2], name_fn=lambda x: f"vtp={x}")
    @parametrize("vpp", [1, 2], name_fn=lambda x: f"vpp={x}")
    @parametrize("vsp", [1, 2], name_fn=lambda x: f"vsp={x}")
    @parametrize("ltp", [1, 2], name_fn=lambda x: f"ltp={x}")
    @parametrize("lpp", [1, 2], name_fn=lambda x: f"lpp={x}")
    @parametrize("lsp", [1, 2], name_fn=lambda x: f"lsp={x}")
    @parametrize(
        "pipeline_schedule",
        ["1f1b", "zbpp"],
        name_fn=lambda x: f"pp={x}",
    )
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        vtp: int,
        vpp: int,
        vsp: int,
        ltp: int,
        lpp: int,
        lsp: int,
        pipeline_schedule: str,
    ):
        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            {"vision": vtp, "llm": ltp},
            {"vision": vpp, "llm": lpp},
            {"vision": vsp, "llm": lsp},
            pipeline_schedule=pipeline_schedule,
        )
