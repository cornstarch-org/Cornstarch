import copy
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.interface import OptimizerWrapper
from torch.optim import Adam
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelModule,
    MultimodalParallelPlugin,
)

from ...test_shardformer.test_multimodal._utils import (
    PolicyTestBase,
    config_class_dict,
    pipeline_template_dict,
)


@contextmanager
def shared_tempdir() -> Iterator[str]:
    """
    Copied from https://github.com/hpcaitech/ColossalAI/blob/v0.4.2/tests/test_checkpoint_io/utils.py
    A temporary directory that is shared across all processes.
    """
    ctx_fn = tempfile.TemporaryDirectory if dist.get_rank() == 0 else nullcontext
    with ctx_fn() as tempdir:
        try:
            obj = [tempdir]
            dist.broadcast_object_list(obj, src=0)
            tempdir = obj[0]  # use the same directory on all ranks
            yield tempdir
        finally:
            dist.barrier()


class TestMultimodalCheckpointIOClass(PolicyTestBase):
    vision_model_class = CLIPVisionModel
    language_model_class = LlamaForCausalLM
    vision_config: CLIPVisionConfig = config_class_dict["clip_vision_model"]
    language_config: LlamaConfig = config_class_dict["llama"]

    @staticmethod
    def data_gen_fn() -> dict:
        microbatch_size = 1
        num_microbatches = 4
        num_batch = microbatch_size * num_microbatches
        input = {
            "pixel_values": torch.rand(num_batch, 3, 224, 224),
            "input_ids": torch.randint(0, 2048, (num_batch, 64)),
            "attention_mask": torch.ones(num_batch, 64),
        }
        input["labels"] = input["input_ids"]

        return input

    @staticmethod
    def loss_fn(outputs: CausalLMOutputWithPast, inputs: Any) -> torch.Tensor:
        return outputs.loss

    def model_fn(self) -> MultimodalModel:
        vision_config = copy.deepcopy(self.vision_config)
        vision_config.pad_token_id = vision_config.eos_token_id
        language_config = copy.deepcopy(self.language_config)
        language_config.pad_token_id = language_config.eos_token_id

        vision_module = self.vision_model_class(vision_config)
        language_module = self.language_model_class(language_config)

        return MultimodalModel(
            encoders={"vision": ModalEncoderModule(vision_module)},
            language_model=language_module,
        )

    def build_model_from_multimodal_plugin(
        self,
        tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
        precision: str,
        mixed: bool,
        test_config: dict[str, Any],
    ) -> tuple[MultimodalParallelModule, OptimizerWrapper, Booster]:
        model = self.model_fn()
        model.to(device=torch.device("cuda"))

        optimizer = Adam(model.parameters(), lr=1e-3)

        vision_plugin = ModalParallelPlugin(
            tp_size=tp_size,
            pipeline_template=pipeline_template_dict[
                (model.vision_encoder.config[0].model_type, vision_pp_size)
            ],
        )
        language_plugin = ModalParallelPlugin(
            tp_size=tp_size,
            pipeline_template=pipeline_template_dict[
                (model.language_model.config.model_type, language_pp_size)
            ],
        )
        plugin = MultimodalParallelPlugin(
            encoder_plugins={"vision": vision_plugin},
            language_model_plugin=language_plugin,
            precision=precision if not mixed else None,
            **test_config,
        )
        if not mixed:
            if precision == "fp16":
                model.to(dtype=torch.float16)
            elif precision == "bf16":
                model.to(dtype=torch.bfloat16)
            else:
                model.to(dtype=torch.float32)

        booster = Booster(plugin=plugin)

        model, optimizer, *_ = booster.boost(model, optimizer, criterion=self.loss_fn)
        return model, optimizer, booster

    def run_forward_backward_with_multimodal_plugin(
        self,
        model: MultimodalParallelModule,
        optimizer: OptimizerWrapper,
        booster: Booster,
    ):
        data = self.data_gen_fn()
        for k, v in data.items():
            data[k] = v.clone().to(device=torch.device("cuda"))

        data_iter = iter([data])
        output = booster.execute_pipeline(
            data_iter,
            model,
            criterion=self.loss_fn,
            optimizer=optimizer,
            return_loss=True,
            return_outputs=False,
        )
        loss = output["loss"]

    @parametrize("shard", [True, False])
    @parametrize(
        "tp, vision_pp, language_pp", [(1, 1, 1), (1, 2, 2), (2, 1, 1), (1, 1, 3)]
    )
    @parametrize(
        "precision, mixed",
        [("fp16", True), ("bf16", False), ("bf16", True), ("fp32", False)],
    )
    def test_state_dict(
        self,
        shard: bool,
        tp: int,
        vision_pp: int,
        language_pp: int,
        precision: str,
        mixed: bool,
    ):
        test_config = {
            "num_microbatches": 4,
            "microbatch_size": 1,
            "initial_scale": 1,
            "enable_flash_attention": True,
        }

        model, optimizer, booster = self.build_model_from_multimodal_plugin(
            tp_size=tp,
            vision_pp_size=vision_pp,
            language_pp_size=language_pp,
            precision=precision,
            mixed=mixed,
            test_config=test_config,
        )

        self.run_forward_backward_with_multimodal_plugin(model, optimizer, booster)
        optimizer.step()
        optimizer.zero_grad()

        with shared_tempdir() as tempdir:
            model_ckpt_path = Path(tempdir) / "model"
            optimizer_ckpt_path = Path(tempdir) / "optimizer"

            booster.save_model(model, model_ckpt_path, shard=shard, size_per_shard=32)
            booster.save_optimizer(
                optimizer, optimizer_ckpt_path, shard=shard, size_per_shard=32
            )
            dist.barrier()


instantiate_parametrized_tests(TestMultimodalCheckpointIOClass)
