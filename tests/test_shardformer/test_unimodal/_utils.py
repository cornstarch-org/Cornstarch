import copy
from contextlib import nullcontext
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.lazy import LazyInitContext
from colossalai.shardformer.policies.auto_policy import _fullname
from torch import nn
from torch.optim import Adam, Optimizer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel

from cornstarch.shardformer.policies.auto_policy import get_autopolicy

from ..utils import (
    ModelClassBase,
    PolicyTestBase,
    all_to_all_gloo,
    all_to_all_single_gloo,
    batch_isend_irecv_gloo,
    check_all_grad_tensors,
    check_loss,
    check_weight,
    get_grad_tensors_for_check,
)


class ColossalaiHybridParallelBase(PolicyTestBase):
    def set_model(self, model: ModelClassBase):
        self.model = model

    def check_fn(
        self,
        booster: Booster,
        org_model: PreTrainedModel,
        sharded_model: ModelWrapper,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: ModelOutput,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ):
        stage_manager = booster.plugin.stage_manager
        # Loss check
        if stage_manager is None or stage_manager.is_last_stage():
            check_loss(
                org_loss, sharded_loss, atol=self.model.atol, rtol=self.model.rtol
            )

        tp_group: dist.ProcessGroup = booster.plugin.tp_group
        sharded_model: PreTrainedModel = sharded_model.unwrap()

        # Gradient check
        grads_to_check = {}

        if stage_manager is None or stage_manager.is_first_stage():
            grads_to_check.update(
                get_grad_tensors_for_check(
                    org_model,
                    sharded_model,
                    self.model.row_layers_to_check,
                    tp_group,
                    atol=self.model.atol,
                    rtol=self.model.rtol,
                    dim=0,
                    verbose=False,
                )
            )
            grads_to_check.update(
                get_grad_tensors_for_check(
                    org_model,
                    sharded_model,
                    self.model.col_layers_to_check,
                    tp_group,
                    atol=self.model.atol,
                    rtol=self.model.rtol,
                    dim=1,
                    verbose=False,
                )
            )
            grads_to_check.update(
                get_grad_tensors_for_check(
                    org_model,
                    sharded_model,
                    self.model.norm_layers_to_check,
                    tp_group,
                    atol=self.model.atol,
                    rtol=self.model.rtol,
                    dim=1,
                    verbose=False,
                )
            )

        # Update parameters
        org_optim.step()
        sharded_optim.step()

        # New parameter check
        if stage_manager is None or stage_manager.is_first_stage():
            check_weight(
                org_model,
                sharded_model,
                self.model.row_layers_to_check,
                tp_group,
                dim=0,
                atol=self.model.atol,
                rtol=self.model.rtol,
            )
            check_weight(
                org_model,
                sharded_model,
                self.model.col_layers_to_check,
                tp_group,
                dim=1,
                atol=self.model.atol,
                rtol=self.model.rtol,
            )

        check_all_grad_tensors(grads_to_check)

    def run_hybrid_parallel(
        self,
        tp_size: int,
        pp_size: int,
        sp_mode: str | None,
        fa: bool,
        precision: str,
    ):
        assert precision in ["bf16", "fp16"]
        precision = torch.bfloat16 if precision == "bf16" else torch.float16

        test_config = dict(
            tp_size=tp_size,
            pp_size=pp_size,
            zero_stage=0,
            num_microbatches=self.num_microbatches,
            microbatch_size=self.microbatch_size,
            initial_scale=1,
            enable_flash_attention=fa,
        )
        if sp_mode is not None:
            test_config.update(
                {
                    "enable_sequence_parallelism": True,
                    "sequence_parallelism_mode": sp_mode,
                    "sp_size": 2,
                }
            )

        (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        ) = self.build_model_from_hybrid_plugin(
            test_config=test_config, precision=precision
        )

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

        with (
            patch.object(
                dist,
                "batch_isend_irecv",
                new=batch_isend_irecv_gloo,
            ),
            patch.object(
                dist,
                "all_to_all",
                new=all_to_all_gloo,
            ),
            patch.object(
                dist,
                "all_to_all_single",
                new=all_to_all_single_gloo,
            ),
            patch(
                "colossalai.pipeline.p2p._check_device",
                return_value=(torch.device("cuda"), False),
            ),
        ):
            org_loss, org_output, sharded_loss, sharded_output = (
                self.run_forward_backward_with_hybrid_plugin(
                    org_model=org_model,
                    sharded_model=sharded_model,
                    sharded_optimizer=sharded_optimizer,
                    criterion=criterion,
                    output_transform_fn=lambda x: x,
                    booster=booster,
                    precision=precision,
                )
            )

        self.check_fn(
            booster=booster,
            org_model=org_model,
            sharded_model=sharded_model,
            org_optim=org_optimizer,
            sharded_optim=sharded_optimizer,
            org_output=org_output,
            sharded_output=sharded_output,
            org_loss=org_loss,
            sharded_loss=sharded_loss,
        )

    def build_model_from_hybrid_plugin(
        self,
        test_config: dict[str, Any],
        precision: torch.dtype,
    ) -> tuple[
        PreTrainedModel,
        Optimizer,
        HybridParallelModule,
        OptimizerWrapper,
        Callable,
        Booster,
    ]:
        use_lazy_init: bool = test_config.pop("use_lazy_init", False)
        use_flash_attention: bool = test_config["enable_flash_attention"]

        ctx = LazyInitContext() if use_lazy_init else nullcontext()
        with ctx:
            org_model = self.model.model_fn(use_flash_attention).to(device="cuda")
            sharded_model = copy.deepcopy(org_model)
        if use_lazy_init:
            ctx.materialize(org_model)

        org_optimizer = Adam(org_model.parameters(), lr=1e-3)
        sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)

        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=get_autopolicy(_fullname(org_model)),
        ):
            plugin = HybridParallelPlugin(**test_config)
            if precision == torch.bfloat16:
                org_model.to(dtype=precision)
                sharded_model.to(dtype=precision)
                plugin.precision = None
            else:
                # Do not cast org_model for fp16 here, as it will be casted in
                # torch.autocast amp
                plugin.precision = "fp16"
            booster = Booster(plugin=plugin)

            sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(
                sharded_model, sharded_optimizer, self.model.loss_fn
            )

        return (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        )

    def run_forward_backward_with_hybrid_plugin(
        self,
        org_model: nn.Module,
        sharded_model: nn.Module,
        sharded_optimizer: Optimizer,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        output_transform_fn: Callable,
        booster: Booster,
        precision: torch.dtype,
    ):
        def _criterion(outputs: BaseModelOutputWithPast, inputs: Any):
            outputs = output_transform_fn(outputs)
            loss = criterion(outputs)
            return loss

        data = self.model.data_gen_fn(self.microbatch_size * self.num_microbatches)

        shard_test_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.to(dtype=precision)
            shard_test_data[k] = v.clone().to("cuda")
        unshard_test_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.to(dtype=precision)
            unshard_test_data[k] = v.clone().to("cuda")

        # use torch.autocast AMP for fp16 training test cases
        org_model.train()
        with torch.autocast(
            device_type="cuda", enabled=precision == torch.float16, dtype=torch.float16
        ):
            org_output = org_model(**unshard_test_data)
            org_loss = criterion(org_output).to(precision)
        org_loss.backward()

        sharded_model.train()
        if booster.plugin.stage_manager is not None:
            data_iter = iter([shard_test_data])
            sharded_output = booster.execute_pipeline(
                data_iter,
                sharded_model,
                _criterion,
                sharded_optimizer,
                return_loss=True,
                return_outputs=False,
            )
            sharded_loss = sharded_output["loss"]
        else:
            sharded_output = sharded_model(**shard_test_data)
            sharded_loss = criterion(sharded_output)
            sharded_optimizer.backward(sharded_loss)

        return org_loss, org_output, sharded_loss, sharded_output
