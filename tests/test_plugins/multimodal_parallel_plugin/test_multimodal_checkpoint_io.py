import copy
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.interface import OptimizerWrapper
from colossalai.testing.comparison import assert_close_loose, check_state_dict_equal
from torch.optim import Adam
from torch.testing._internal.common_distributed import TEST_SKIPS
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
            precision=precision if mixed else None,
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

        optimizer = Adam(model.parameters(), lr=1e-3)
        model, optimizer, *_ = booster.boost(model, optimizer, criterion=self.loss_fn)
        return model, optimizer, booster

    def run_forward_backward_with_multimodal_plugin(
        self,
        model: MultimodalParallelModule,
        optimizer: OptimizerWrapper,
        booster: Booster,
        data: Optional[dict[str, torch.Tensor]] = None,
    ):
        if data is None:
            data = self.data_gen_fn()
            for k, v in data.items():
                data[k] = v.clone().to(device=torch.device("cuda"))

        data_iter = iter([data])
        booster.execute_pipeline(
            data_iter,
            model,
            criterion=self.loss_fn,
            optimizer=optimizer,
            return_loss=True,
            return_outputs=False,
        )

    @parametrize("shard", [True, False])
    @parametrize(
        "tp, vision_pp, language_pp", [(1, 1, 1), (1, 2, 2), (2, 1, 1), (1, 1, 3)]
    )
    @parametrize(
        "precision, mixed",
        [("fp16", True), ("bf16", False), ("bf16", True), ("fp32", False)],
        name_fn=lambda p, m: f"precision_{p}_mixed" if m else f"precision_{p}",
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
        if shard is False:
            sys.exit(TEST_SKIPS["generic"].exit_code)

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

            new_model, new_optimizer, booster = self.build_model_from_multimodal_plugin(
                tp_size=tp,
                vision_pp_size=vision_pp,
                language_pp_size=language_pp,
                precision=precision,
                mixed=mixed,
                test_config=test_config,
            )

            booster.load_model(
                new_model,
                {
                    "language_model": model_ckpt_path / "language_model",
                    "vision_encoder.module": model_ckpt_path
                    / "vision_encoder"
                    / "module",
                    "vision_encoder.projector": model_ckpt_path
                    / "vision_encoder"
                    / "projector",
                },
            )
            check_state_dict_equal(
                model.unwrap().state_dict(), new_model.unwrap().state_dict()
            )
            booster.load_optimizer(
                new_optimizer,
                {
                    "language_model": optimizer_ckpt_path / "language_model",
                    "vision_encoder": optimizer_ckpt_path / "vision_encoder",
                },
            )
            check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())

            dist.barrier()

        # Check whether the loaded model & optimizer works well.
        data = self.data_gen_fn()
        for k, v in data.items():
            data[k] = v.clone().to(device=torch.device("cuda"))

        self.reset_seed()
        self.run_forward_backward_with_multimodal_plugin(
            model, optimizer, booster, copy.deepcopy(data)
        )
        self.reset_seed()
        self.run_forward_backward_with_multimodal_plugin(
            new_model, new_optimizer, booster, copy.deepcopy(data)
        )

        optimizer.step()
        new_optimizer.step()

        # Check updated weights.
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert_close_loose(p1, p2, atol=5e-3, rtol=5e-3)

        dist.barrier()

    @parametrize(
        "tp_size, vision_pp, language_pp",
        [(1, 1, 1), (1, 2, 2), (2, 1, 1), (1, 1, 3)],
        name_fn=lambda tp, vp, lp: f"tp_{tp}_pp_({vp},{lp})",
    )
    @parametrize("language_model_frozen", [True, False], name_fn=lambda f: f"lf_{f}")
    @parametrize("vision_encoder_frozen", [True, False], name_fn=lambda f: f"vf_{f}")
    def test_frozen_model_not_checkpointed(
        self,
        tp_size: int,
        vision_pp: int,
        language_pp: int,
        language_model_frozen: bool,
        vision_encoder_frozen: bool,
    ):
        test_config = {
            "num_microbatches": 4,
            "microbatch_size": 1,
        }

        model, _, booster = self.build_model_from_multimodal_plugin(
            tp_size=tp_size,
            vision_pp_size=vision_pp,
            language_pp_size=language_pp,
            precision="bf16",
            mixed=False,
            test_config=test_config,
        )

        # Freeze portion of model
        model.module.language_model.train(mode=not language_model_frozen)
        model.module.vision_encoder.train(
            module=not vision_encoder_frozen, projector=True
        )

        # Save model
        with shared_tempdir() as tempdir:
            model_ckpt_path = Path(tempdir) / "model"
            booster.save_model(model, model_ckpt_path, shard=True)

            dist.barrier()

            assert (
                not language_model_frozen
                == (model_ckpt_path / "language_model").exists()
            )
            assert (model_ckpt_path / "vision_encoder").exists()
            assert (
                not vision_encoder_frozen
                == (model_ckpt_path / "vision_encoder" / "module").exists()
            )
            assert (model_ckpt_path / "vision_encoder" / "projector").exists()

            dist.barrier()


instantiate_parametrized_tests(TestMultimodalCheckpointIOClass)
