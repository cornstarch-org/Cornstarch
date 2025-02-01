import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.lazy import LazyInitContext, LazyTensor
from colossalai.testing.comparison import (
    assert_close_loose,
    assert_not_equal,
    check_state_dict_equal,
)
from torch.testing._internal.common_distributed import TEST_SKIPS
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalParallelModule,
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
        for modal_key in self.encoders.keys():
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

    @parametrize("shard", [True])
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

    @parametrize("tp_size", [1, 2])
    @parametrize("vision_pp_size, language_pp_size", [(1, 1), (2, 2), (1, 3)])
    @parametrize("vision_encoder_frozen", [True, False], name_fn=lambda f: f"vf_{f}")
    @parametrize("language_model_frozen", [True, False], name_fn=lambda f: f"lf_{f}")
    def test_frozen_params_not_checkpointed(
        self,
        tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
        vision_encoder_frozen: bool,
        language_model_frozen: bool,
    ):
        self.set_model(
            encoders={"vision": CLIPModelBase()},
            llm=LlamaForCausalLMBase(),
        )

        model, optimizer, criterion, booster = self.build_model_from_multimodal_plugin(
            tp_size, {"vision": vision_pp_size, "llm": language_pp_size}
        )

        # Free portion of the model
        model.train(
            encoders_mode={"vision": (not vision_encoder_frozen, True)},
            llm_mode=(not language_model_frozen),
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

    @parametrize("tp_size", [1, 2])
    @parametrize("vision_pp_size, language_pp_size", [(1, 1), (2, 2), (1, 3)])
    @parametrize("load_projector", [True, False], name_fn=lambda f: f"lp_{f}")
    def test_load_lazy_init_model(
        self,
        tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
        load_projector: bool,
    ):
        self.set_model(
            encoders={"vision": CLIPModelBase()},
            llm=LlamaForCausalLMBase(),
        )

        model, _, criterion, booster = self.build_model_from_multimodal_plugin(
            tp_size, {"vision": vision_pp_size, "llm": language_pp_size}
        )
        model.train()

        with shared_tempdir() as tempdir:
            model_ckpt_path = Path(tempdir) / "model"
            booster.save_model(model, model_ckpt_path, shard=True, size_per_shard=32)

            dist.barrier()

            language_model_ckpth_path = model_ckpt_path / "language_model"
            vision_encoder_ckpth_path = model_ckpt_path / "vision_encoder" / "module"
            vision_projector_ckpt_path = (
                model_ckpt_path / "vision_encoder" / "projector"
            )

            with LazyInitContext():
                lazy_model = self.build_model_from_pretrained(
                    encoder_paths={
                        "vision": (
                            vision_encoder_ckpth_path,
                            vision_projector_ckpt_path if load_projector else None,
                        )
                    },
                    language_model_path=language_model_ckpth_path,
                )

            assert all(isinstance(p, LazyTensor) for p in lazy_model.parameters())

            test_config = dict(
                num_microbatches=self.num_microbatches,
                microbatch_size=self.microbatch_size,
                initial_scale=1,
                enable_flash_attention=True,
                precision=torch.bfloat16,
            )
            lazy_model, _, _, booster = self.parallelize_model(
                lazy_model,
                tp_size,
                {"vision": vision_pp_size, "llm": language_pp_size},
                None,
                test_config,
                torch.bfloat16,
            )

            assert all(not isinstance(p, LazyTensor) for p in lazy_model.parameters())

            booster.load_model(lazy_model, checkpoint=None, strict=False)

            dist.barrier()

        org_language_params = dict(model.module.language_model.named_parameters())
        lazy_language_params = dict(lazy_model.module.language_model.named_parameters())

        assert list(org_language_params.keys()) == list(lazy_language_params.keys())
        for k in org_language_params.keys():
            assert_close_loose(
                org_language_params[k], lazy_language_params[k], atol=1e-3
            )

        org_vision_params = dict(model.module.vision_encoder.module.named_parameters())
        lazy_vision_params = dict(
            lazy_model.module.vision_encoder.module.named_parameters()
        )

        assert list(org_vision_params.keys()) == list(lazy_vision_params.keys())
        for k in org_vision_params.keys():
            assert_close_loose(org_vision_params[k], lazy_vision_params[k], atol=1e-3)

        org_projector_params = dict(
            model.module.vision_encoder.projector.named_parameters()
        )
        lazy_projector_params = dict(
            lazy_model.module.vision_encoder.projector.named_parameters()
        )

        assert list(org_vision_params.keys()) == list(lazy_vision_params.keys())
        if load_projector:
            for k in org_projector_params.keys():
                assert_close_loose(
                    org_projector_params[k], lazy_projector_params[k], atol=1e-3
                )
        else:
            for k in org_projector_params.keys():
                assert_not_equal(org_projector_params[k], lazy_projector_params[k])
