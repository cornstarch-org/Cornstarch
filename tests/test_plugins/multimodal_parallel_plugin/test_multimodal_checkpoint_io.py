import copy
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.lazy import LazyInitContext, LazyTensor
from colossalai.testing.comparison import assert_close_loose, check_state_dict_equal
from torch.optim import Adam, Optimizer
from torch.testing._internal.common_distributed import TEST_SKIPS
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProjector,
)
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelModule,
    MultimodalParallelPlugin,
)

from ...test_shardformer.model_zoo import CLIPModelBase, LlamaForCausalLMBase
from ...test_shardformer.utils import CornstarchMultimodalParallelBase


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


class CornstarchCheckpointTestBase(CornstarchMultimodalParallelBase):
    def check_fn(self, *args, **kwargs):
        raise NotImplementedError("Should not be called")

    @property
    def world_size(self) -> int:
        return 16

    def postprocess_data_for_sharded_model(
        self,
        data: dict,
        precision: torch.dtype,
    ):
        """
        Inject encoder tokens to the input_ids for the multimodal model.
        """
        data = super().postprocess_data_for_sharded_model(data, precision)

        input_ids: torch.Tensor = data["input_ids"]
        encoder_tokens: list[torch.Tensor] = []
        for modal_key, model_class_base in self.encoders.items():
            # num_encoder_tokens is a list[int] type, a list of number of tokens for each batch.
            # Implement a 2D tensor with the shape of (batch_size, num_encoder_tokens)
            encoder_tokens.append(
                torch.full(
                    (input_ids.shape[0], 32),
                    fill_value=self.token_ids[modal_key],
                    dtype=torch.long,
                    device=input_ids.device,
                )
            )

        # prepend it to input_ids
        input_ids = torch.cat(encoder_tokens + [input_ids], dim=1)
        data["input_ids"] = input_ids
        data["labels"] = input_ids

        return data

    def build_model_from_multimodal_plugin(
        self, tp_size: int, module_pp_size: dict[str, int]
    ) -> tuple[MultimodalParallelModule, OptimizerWrapper, Callable, Booster]:
        test_config = dict(
            num_microbatches=self.num_microbatches,
            microbatch_size=self.microbatch_size,
            initial_scale=1,
            enable_flash_attention=True,
            precision=torch.bfloat16,
        )

        (
            _,
            _,
            model,
            optimizer,
            criterion,
            booster,
        ) = super().build_model_from_multimodal_plugin(
            tp_size, module_pp_size, None, test_config, torch.bfloat16
        )

        return model, optimizer, criterion, booster

    def run_forward_backward_with_multimodal_plugin(
        self,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        criterion: Callable,
        booster: Booster,
    ) -> None:
        super().run_forward_backward_with_multimodal_plugin(
            org_model=None,
            sharded_model=model,
            sharded_optimizer=optimizer,
            criterion=criterion,
            output_transform_fn=lambda x: x,
            booster=booster,
            precision=torch.bfloat16,
            run_original_model=False,
            run_sharded_model=True,
        )


@instantiate_parametrized_tests
class MultimodalCheckpointIOClass(CornstarchCheckpointTestBase):

    @parametrize("shard", [True, False])
    @parametrize("tp_size", [1, 2])
    @parametrize("vision_pp_size, language_pp_size", [(1, 1), (2, 2), (1, 3)])
    def test_state_dict(
        self,
        shard: bool,
        tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
    ):
        if shard is False:
            sys.exit(TEST_SKIPS["generic"].exit_code)

        self.set_model(
            encoders={"vision": CLIPModelBase()},
            llm=LlamaForCausalLMBase(),
        )

        model, optimizer, criterion, booster = self.build_model_from_multimodal_plugin(
            tp_size, {"vision": vision_pp_size, "llm": language_pp_size}
        )

        self.run_forward_backward_with_multimodal_plugin(
            model, optimizer, criterion, booster
        )

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

            new_model, new_optimizer, _, booster = (
                self.build_model_from_multimodal_plugin(
                    tp_size, {"vision": vision_pp_size, "llm": language_pp_size}
                )
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

            booster.load_optimizer(
                new_optimizer,
                {
                    "language_model": optimizer_ckpt_path / "language_model",
                    "vision_encoder": optimizer_ckpt_path / "vision_encoder",
                },
            )

            dist.barrier()

        for name, p1 in model.named_parameters():
            if p1.data is None:
                continue
            p2 = new_model.get_parameter(name)
            assert_close_loose(p1, p2, atol=5e-3, rtol=5e-3)
        check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())

        # Check whether the loaded model & optimizer works well.
        self.reset_seed()
        self.run_forward_backward_with_multimodal_plugin(
            model, optimizer, criterion, booster
        )
        self.reset_seed()
        self.run_forward_backward_with_multimodal_plugin(
            new_model, new_optimizer, criterion, booster
        )

        optimizer.step()
        new_optimizer.step()

        # Check updated weights.
        for name, p1 in model.named_parameters():
            if p1.data is None:
                continue
            p2 = new_model.get_parameter(name)
            assert_close_loose(p1, p2, atol=5e-3, rtol=5e-3)


# class TestMultimodalCheckpointIOClass(PolicyTestBase):
#     vision_model_class = CLIPVisionModel
#     language_model_class = LlamaForCausalLM
#     vision_config: CLIPVisionConfig = clip_config
#     language_config: LlamaConfig = llama_config

#     def data_gen_fn(self) -> dict:
#         microbatch_size = 1
#         num_microbatches = 4
#         num_batch = microbatch_size * num_microbatches
#         input = {
#             "pixel_values": torch.rand(num_batch, 3, 224, 224),
#             "input_ids": torch.randint(0, 2048, (num_batch, 64)),
#             "attention_mask": torch.ones(num_batch, 64),
#         }
#         input["labels"] = input["input_ids"]

#         return input

#     def check_fn(self, *args, **kwargs):
#         pass

#     @staticmethod
#     def loss_fn(outputs: CausalLMOutputWithPast) -> torch.Tensor:
#         return outputs.loss

#     def model_fn(self) -> MultimodalModel:
#         vision_config = copy.deepcopy(self.vision_config)
#         vision_config.pad_token_id = vision_config.eos_token_id
#         language_config = copy.deepcopy(self.language_config)
#         language_config.pad_token_id = language_config.eos_token_id

#         vision_module = self.vision_model_class(vision_config)
#         language_module = self.language_model_class(language_config)

#         return MultimodalModel(
#             encoders={"vision": ModalEncoderModule(vision_module)},
#             language_model=language_module,
#         )

#     def build_model_from_multimodal_plugin(
#         self,
#         tp_size: int,
#         vision_pp_size: int,
#         language_pp_size: int,
#         precision: str,
#         mixed: bool,
#         test_config: dict[str, Any],
#         model: Optional[MultimodalModel] = None,
#     ) -> tuple[MultimodalParallelModule, OptimizerWrapper, Booster]:
#         if model is None:
#             model = self.model_fn()
#             model.to(device=torch.device("cuda"))

#         vision_plugin = ModalParallelPlugin(
#             tp_size=tp_size,
#             pipeline_template=CornstarchMultimodalParallelBase.get_pipeline_template(
#                 model.vision_encoder, vision_pp_size
#             ),
#         )
#         language_plugin = ModalParallelPlugin(
#             tp_size=tp_size,
#             pipeline_template=CornstarchMultimodalParallelBase.get_pipeline_template(
#                 model.language_model, language_pp_size
#             ),
#         )
#         plugin = MultimodalParallelPlugin(
#             encoder_plugins={"vision": vision_plugin},
#             language_model_plugin=language_plugin,
#             precision=precision if mixed else None,
#             **test_config,
#         )
#         if not mixed:
#             if precision == "fp16":
#                 model.to(dtype=torch.float16)
#             elif precision == "bf16":
#                 model.to(dtype=torch.bfloat16)
#             else:
#                 model.to(dtype=torch.float32)

#         booster = Booster(plugin=plugin)

#         optimizer = Adam(model.parameters(), lr=1e-3)
#         model, optimizer, *_ = booster.boost(model, optimizer, criterion=self.loss_fn)
#         return model, optimizer, booster

#     def run_forward_backward_with_multimodal_plugin(
#         self,
#         model: MultimodalParallelModule,
#         optimizer: OptimizerWrapper,
#         booster: Booster,
#         data: Optional[dict[str, torch.Tensor]] = None,
#     ):
#         if data is None:
#             data = self.data_gen_fn()
#             for k, v in data.items():
#                 data[k] = v.clone().to(device=torch.device("cuda"))

#         data_iter = iter([data])
#         booster.execute_pipeline(
#             data_iter,
#             model,
#             criterion=self.loss_fn,
#             optimizer=optimizer,
#             return_loss=True,
#             return_outputs=False,
#         )

#     @parametrize("shard", [True, False])
#     @parametrize(
#         "tp, vision_pp, language_pp", [(1, 1, 1), (1, 2, 2), (2, 1, 1), (1, 1, 3)]
#     )
#     @parametrize(
#         "precision, mixed",
#         [("fp16", True), ("bf16", False), ("bf16", True), ("fp32", False)],
#         name_fn=lambda p, m: f"precision_{p}_mixed" if m else f"precision_{p}",
#     )
#     def test_state_dict(
#         self,
#         shard: bool,
#         tp: int,
#         vision_pp: int,
#         language_pp: int,
#         precision: str,
#         mixed: bool,
#     ):
#         if shard is False:
#             sys.exit(TEST_SKIPS["generic"].exit_code)

#         test_config = {
#             "num_microbatches": 4,
#             "microbatch_size": 1,
#             "initial_scale": 1,
#             "enable_flash_attention": True,
#         }

#         model, optimizer, booster = self.build_model_from_multimodal_plugin(
#             tp_size=tp,
#             vision_pp_size=vision_pp,
#             language_pp_size=language_pp,
#             precision=precision,
#             mixed=mixed,
#             test_config=test_config,
#         )

#         self.run_forward_backward_with_multimodal_plugin(model, optimizer, booster)
#         optimizer.step()
#         optimizer.zero_grad()

#         with shared_tempdir() as tempdir:
#             model_ckpt_path = Path(tempdir) / "model"
#             optimizer_ckpt_path = Path(tempdir) / "optimizer"

#             booster.save_model(model, model_ckpt_path, shard=shard, size_per_shard=32)
#             booster.save_optimizer(
#                 optimizer, optimizer_ckpt_path, shard=shard, size_per_shard=32
#             )
#             dist.barrier()

#             new_model, new_optimizer, booster = self.build_model_from_multimodal_plugin(
#                 tp_size=tp,
#                 vision_pp_size=vision_pp,
#                 language_pp_size=language_pp,
#                 precision=precision,
#                 mixed=mixed,
#                 test_config=test_config,
#             )

#             booster.load_model(
#                 new_model,
#                 {
#                     "language_model": model_ckpt_path / "language_model",
#                     "vision_encoder.module": model_ckpt_path
#                     / "vision_encoder"
#                     / "module",
#                     "vision_encoder.projector": model_ckpt_path
#                     / "vision_encoder"
#                     / "projector",
#                 },
#             )
#             check_state_dict_equal(
#                 model.unwrap().state_dict(), new_model.unwrap().state_dict()
#             )
#             booster.load_optimizer(
#                 new_optimizer,
#                 {
#                     "language_model": optimizer_ckpt_path / "language_model",
#                     "vision_encoder": optimizer_ckpt_path / "vision_encoder",
#                 },
#             )
#             check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())

#             dist.barrier()

#         # Check whether the loaded model & optimizer works well.
#         data = self.data_gen_fn()
#         for k, v in data.items():
#             data[k] = v.clone().to(device=torch.device("cuda"))

#         self.reset_seed()
#         self.run_forward_backward_with_multimodal_plugin(
#             model, optimizer, booster, copy.deepcopy(data)
#         )
#         self.reset_seed()
#         self.run_forward_backward_with_multimodal_plugin(
#             new_model, new_optimizer, booster, copy.deepcopy(data)
#         )

#         optimizer.step()
#         new_optimizer.step()

#         # Check updated weights.
#         for p1, p2 in zip(model.parameters(), new_model.parameters()):
#             assert_close_loose(p1, p2, atol=5e-3, rtol=5e-3)

#         dist.barrier()

#     @parametrize(
#         "tp_size, vision_pp, language_pp",
#         [(1, 1, 1), (1, 2, 2), (2, 1, 1), (1, 1, 3)],
#         name_fn=lambda tp, vp, lp: f"tp_{tp}_pp_({vp},{lp})",
#     )
#     @parametrize("language_model_frozen", [True, False], name_fn=lambda f: f"lf_{f}")
#     @parametrize("vision_encoder_frozen", [True, False], name_fn=lambda f: f"vf_{f}")
#     def test_frozen_model_not_checkpointed(
#         self,
#         tp_size: int,
#         vision_pp: int,
#         language_pp: int,
#         language_model_frozen: bool,
#         vision_encoder_frozen: bool,
#     ):
#         test_config = {
#             "num_microbatches": 4,
#             "microbatch_size": 1,
#         }

#         model, _, booster = self.build_model_from_multimodal_plugin(
#             tp_size=tp_size,
#             vision_pp_size=vision_pp,
#             language_pp_size=language_pp,
#             precision="bf16",
#             mixed=False,
#             test_config=test_config,
#         )

#         # Freeze portion of model
#         model.module.language_model.train(mode=not language_model_frozen)
#         model.module.vision_encoder.train(
#             module=not vision_encoder_frozen, projector=True
#         )

#         # Save model
#         with shared_tempdir() as tempdir:
#             model_ckpt_path = Path(tempdir) / "model"
#             booster.save_model(model, model_ckpt_path, shard=True)

#             dist.barrier()

#             assert (
#                 not language_model_frozen
#                 == (model_ckpt_path / "language_model").exists()
#             )
#             assert (model_ckpt_path / "vision_encoder").exists()
#             assert (
#                 not vision_encoder_frozen
#                 == (model_ckpt_path / "vision_encoder" / "module").exists()
#             )
#             assert (model_ckpt_path / "vision_encoder" / "projector").exists()

#             dist.barrier()

#     @parametrize(
#         "tp_size, vision_pp, language_pp",
#         [(1, 1, 1), (1, 2, 2), (2, 1, 1), (1, 1, 3)],
#         name_fn=lambda tp, vp, lp: f"tp_{tp}_pp_({vp},{lp})",
#     )
#     def test_load_lazy_init_model(self, tp_size: int, vision_pp: int, language_pp: int):

#         test_config = {
#             "num_microbatches": 4,
#             "microbatch_size": 1,
#         }

#         org_model, _, booster = self.build_model_from_multimodal_plugin(
#             tp_size=tp_size,
#             vision_pp_size=vision_pp,
#             language_pp_size=language_pp,
#             precision="bf16",
#             mixed=False,
#             test_config=test_config,
#         )
#         org_model.module.language_model.train(mode=True)
#         org_model.module.vision_encoder.train(module=True, projector=True)

#         with shared_tempdir() as tempdir:
#             model_ckpt_path = Path(tempdir) / "model"
#             booster.save_model(
#                 org_model, model_ckpt_path, shard=True, size_per_shard=32
#             )

#             dist.barrier()

#             language_model_ckpth_path = model_ckpt_path / "language_model"
#             vision_encoder_ckpth_path = model_ckpt_path / "vision_encoder" / "module"
#             vision_projector_ckpt_path = (
#                 model_ckpt_path / "vision_encoder" / "projector"
#             )

#             with LazyInitContext():
#                 vision_model = self.vision_model_class.from_pretrained(
#                     vision_encoder_ckpth_path, torch_dtype=torch.bfloat16
#                 )
#                 vision_projector = MultimodalProjector.from_pretrained(
#                     vision_projector_ckpt_path, torch_dtype=torch.bfloat16
#                 )
#                 language_model = self.language_model_class.from_pretrained(
#                     language_model_ckpth_path, torch_dtype=torch.bfloat16
#                 )
#                 lazy_model = MultimodalModel(
#                     encoders={
#                         "vision": ModalEncoderModule(vision_model, vision_projector)
#                     },
#                     language_model=language_model,
#                 )

#             lazy_model, _, booster = self.build_model_from_multimodal_plugin(
#                 tp_size=tp_size,
#                 vision_pp_size=vision_pp,
#                 language_pp_size=language_pp,
#                 precision="bf16",
#                 mixed=False,
#                 test_config=test_config,
#                 model=lazy_model,
#             )

#             assert all(not isinstance(p, LazyTensor) for p in lazy_model.parameters())

#             booster.load_model(
#                 lazy_model,
#                 {
#                     "language_model": lazy_model.module.language_model._pretrained,
#                     "vision_encoder.module": lazy_model.module.vision_encoder.module._pretrained,
#                     "vision_encoder.projector": lazy_model.module.vision_encoder.projector._pretrained,
#                 },
#             )

#             dist.barrier()

#         org_language_params = dict(org_model.module.language_model.named_parameters())
#         lazy_language_params = dict(lazy_model.module.language_model.named_parameters())

#         assert list(org_language_params.keys()) == list(lazy_language_params.keys())
#         for k in org_language_params.keys():
#             assert_close_loose(
#                 org_language_params[k], lazy_language_params[k], atol=1e-3
#             )

#         org_vision_params = dict(
#             org_model.module.vision_encoder.module.named_parameters()
#         )
#         lazy_vision_params = dict(
#             lazy_model.module.vision_encoder.module.named_parameters()
#         )

#         assert list(org_vision_params.keys()) == list(lazy_vision_params.keys())
#         for k in org_vision_params.keys():
#             assert_close_loose(org_vision_params[k], lazy_vision_params[k], atol=1e-3)

#         # Code below is a test that uses actual pretrained model from HF hub.
#         # test_config = {
#         #     "num_microbatches": 4,
#         #     "microbatch_size": 1,
#         # }

#         # def model_fn(tempdir: str) -> MultimodalModel:
#         #     vision_model = self.vision_model_class.from_pretrained(
#         #         "openai/clip-vit-base-patch16", cache_dir=tempdir
#         #     )
#         #     language_model = self.language_model_class.from_pretrained(
#         #         "meta-llama/Meta-Llama-3.1-8B", cache_dir=tempdir
#         #     )

#         #     return MultimodalModel(
#         #         encoders={"vision": ModalEncoderModule(vision_model)},
#         #         language_model=language_model,
#         #     )

#         # with shared_tempdir() as tempdir:
#         #     org_model = model_fn(tempdir)
#         #     assert all(not isinstance(p, LazyTensor) for p in org_model.parameters())

#         #     org_model, _, booster = self.build_model_from_multimodal_plugin(
#         #         tp_size=tp_size,
#         #         vision_pp_size=vision_pp,
#         #         language_pp_size=language_pp,
#         #         precision="bf16",
#         #         mixed=False,
#         #         test_config=test_config,
#         #         model=org_model,
#         #     )
#         #     assert all(not isinstance(p, LazyTensor) for p in org_model.parameters())

#         #     with LazyInitContext():
#         #         lazy_model = model_fn(tempdir)
#         #         assert all(isinstance(p, LazyTensor) for p in lazy_model.parameters())

#         #         lazy_model, _, booster = self.build_model_from_multimodal_plugin(
#         #             tp_size=tp_size,
#         #             vision_pp_size=vision_pp,
#         #             language_pp_size=language_pp,
#         #             precision="bf16",
#         #             mixed=False,
#         #             test_config=test_config,
#         #             model=lazy_model,
#         #         )
#         #         assert all(
#         #             not isinstance(p, LazyTensor) for p in lazy_model.parameters()
#         #         )

#         #     booster.load_model(
#         #         lazy_model,
#         #         {
#         #             "language_model": lazy_model.module.language_model._pretrained,
#         #             "vision_encoder.module": lazy_model.module.vision_encoder.module._pretrained,
#         #         },
#         #         strict=False,  # projector will not be loaded
#         #     )

#         # org_language_params = dict(org_model.module.language_model.named_parameters())
#         # lazy_language_params = dict(lazy_model.module.language_model.named_parameters())

#         # assert list(org_language_params.keys()) == list(lazy_language_params.keys())
#         # for k in org_language_params.keys():
#         #     assert_close_loose(
#         #         org_language_params[k], lazy_language_params[k], atol=1e-3
#         #     )

#         # org_vision_params = dict(
#         #     org_model.module.vision_encoder.module.named_parameters()
#         # )
#         # lazy_vision_params = dict(
#         #     lazy_model.module.vision_encoder.module.named_parameters()
#         # )

#         # assert list(org_vision_params.keys()) == list(lazy_vision_params.keys())
#         # for k in org_vision_params.keys():
#         #     assert_close_loose(org_vision_params[k], lazy_vision_params[k], atol=1e-3)


# instantiate_parametrized_tests(TestMultimodalCheckpointIOClass)
