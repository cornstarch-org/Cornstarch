from typing import Type

import pytest
import torch.distributed as dist
from colossalai.device import device_mesh
from pytest_mock import MockerFixture
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM
from transformers.models.opt import OPTConfig, OPTForCausalLM
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

from cornstarch.pipeline_template import PipelineTemplate


class Module(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(num_layers)])
        self.start_idx, self.end_idx = 0, num_layers

    def forward(self, x):
        for layer in self.layer[self.start_idx : self.end_idx]:
            x = layer(x)
        return x


encoder1_template = PipelineTemplate(
    "encoder1", [["layer.0", "layer.1"], ["layer.2", "layer.3"]]
)
encoder2_template = PipelineTemplate(
    "encoder2", [["layer.0", "layer.1"], ["layer.2", "layer.3"], ["layer.4", "layer.5"]]
)
encoder3_template = PipelineTemplate(
    "encoder3", [["layer.0", "layer.1", "layer.2"], ["layer.3", "layer.4"]]
)

llm_template_2stages = PipelineTemplate(
    "llm", [["layer.0", "layer.1", "layer.2"], ["layer.3", "layer.4"]]
)
llm_template_4stages = PipelineTemplate(
    "llm",
    [
        ["layer.0", "layer.1", "layer.2"],
        ["layer.3"],
        ["layer.4", "layer.5", "layer.6", "layer.7"],
        ["layer.8", "layer.9"],
    ],
)


whisper_config: WhisperConfig = WhisperConfig.from_pretrained("openai/whisper-small")
whisper_config.encoder_layers = 2
whisper_config._attn_implementation = "eager"
audio_configs = [("openai/whisper-small", whisper_config, WhisperEncoder)]

expected_audio_module_layers = {
    "openai/whisper-small": [
        "module.conv1",
        "module.conv2",
        "module.embed_positions",
        "module.layers.0",
        "module.layers.1",
        "module.layer_norm",
        "projector.projection",
    ]
}

clip_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
vision_configs = [("openai/clip-vit-base-patch32", clip_config, CLIPVisionModel)]
for _, config, _ in vision_configs:
    config.num_hidden_layers = 2
    config._attn_implementation = "eager"

expected_vision_module_layers = {
    "openai/clip-vit-base-patch32": [
        "module.vision_model.embeddings",
        "module.vision_model.pre_layrnorm",
        "module.vision_model.encoder.layers.0",
        "module.vision_model.encoder.layers.1",
        "module.vision_model.post_layernorm",
        "projector.projection",
    ]
}

mistral_config = MistralConfig.from_pretrained("mistralai/Mistral-7B-v0.3")
llama_config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
opt_config = OPTConfig.from_pretrained("facebook/opt-125m")

language_configs: list[tuple[str, PretrainedConfig, Type[PreTrainedModel]]] = [
    ("mistralai/Mistral-7B-v0.3", mistral_config, MistralForCausalLM),
    ("meta-llama/Meta-Llama-3-8B", llama_config, LlamaForCausalLM),
    ("facebook/opt-125m", opt_config, OPTForCausalLM),
]
for _, config, _ in language_configs:
    config.num_hidden_layers = 3
    config.hidden_size = 256
    config.intermediate_size = 256
    config.num_attention_heads = 8
    if hasattr(config, "word_embed_proj_dim"):  # for opt
        config.word_embed_proj_dim = 256
    config._attn_implementation = "eager"
    config.tie_word_embeddings = False  # opt uses it


expected_language_module_layers = {
    "mistralai/Mistral-7B-v0.3": [
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.norm",
        "lm_head",
    ],
    "meta-llama/Meta-Llama-3-8B": [
        "model.embed_tokens",
        "model.rotary_emb",
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.norm",
        "lm_head",
    ],
    "facebook/opt-125m": [
        "model.decoder.embed_tokens",
        "model.decoder.embed_positions",
        "model.decoder.layers.0",
        "model.decoder.layers.1",
        "model.decoder.layers.2",
        "model.decoder.final_layer_norm",
        "lm_head",
    ],
}

expected_audio_module_layers_per_stage = {
    # 2-stage expected layer distribution
    "openai/whisper-small": [
        [
            "module.conv1",
            "module.conv2",
            "module.embed_positions",
            "module.layers.0",
        ],
        [
            "module.layers.1",
            "module.layer_norm",
            "projector.projection",
        ],
    ]
}

expected_vision_module_layers_per_stage = {
    # 2-stage expected layer distribution
    "openai/clip-vit-base-patch32": [
        [
            "module.vision_model.embeddings",
            "module.vision_model.pre_layrnorm",
            "module.vision_model.encoder.layers.0",
        ],
        [
            "module.vision_model.encoder.layers.1",
            "module.vision_model.post_layernorm",
            "projector.projection",
        ],
    ]
}

expected_language_module_layers_per_stage = {
    # 3-stage expected layer distribution
    "mistralai/Mistral-7B-v0.3": [
        ["model.embed_tokens", "model.layers.0"],
        ["model.layers.1"],
        ["model.layers.2", "model.norm", "lm_head"],
    ],
    "meta-llama/Meta-Llama-3-8B": [
        ["model.embed_tokens", "model.layers.0"],
        ["model.layers.1"],
        ["model.layers.2", "model.norm", "lm_head"],
    ],
    "facebook/opt-125m": [
        [
            "model.decoder.embed_tokens",
            "model.decoder.embed_positions",
            "model.decoder.layers.0",
        ],
        ["model.decoder.layers.1"],
        ["model.decoder.layers.2", "model.decoder.final_layer_norm", "lm_head"],
    ],
}


class TestPluginInitializationWithFakeBackendBase:
    @pytest.fixture(autouse=True)
    def fake_backend(self, mocker: MockerFixture):
        mocker.patch.object(
            device_mesh.DeviceMesh,
            "_DIST_BACKEND",
            {"cuda": "nccl", "cpu": "gloo", "npu": "hccl", None: "fake"},
        )
        yield

        if dist.is_initialized():
            dist.destroy_process_group()

    def check_layers_cover_all_params(
        self, layer_names: list[str], param_names: list[str]
    ):
        used_prefixes = set()
        for param_name in param_names:
            covered = False
            for layer_name in layer_names:
                if param_name.startswith(layer_name):
                    used_prefixes.add(layer_name)
                    covered = True
                    break
            if not covered:
                return False

        return sorted(used_prefixes) == sorted(set(layer_names))
