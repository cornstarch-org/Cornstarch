import copy
import os
import random
import sys
# import unittest
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, Optional
# from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.lazy import LazyInitContext
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer._utils import getattr_
from colossalai.shardformer.layer.qkv_fused_linear import (
    FusedLinear1D_Col,
    gather_fused_qkv_in_gpt2_style,
)
from colossalai.shardformer.policies.auto_policy import _fullname
from colossalai.tensor.d_tensor.api import (
    is_customized_distributed_tensor,
    is_distributed_tensor,
)
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
# from torch.testing import assert_close
# from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
# from torch.testing._internal.common_utils import FILE_SCHEMA
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from cornstarch.shardformer.policies.auto_policy import get_autopolicy

import torch
from colossalai.booster import Booster
from colossalai.interface import ModelWrapper, OptimizerWrapper
from torch.optim import Optimizer

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    LlamaPreTrainedModel,
)

from .utils import (
    check_all_grad_tensors,
    check_loss,
    check_weight,
    get_grad_tensors_for_check,
    unwrap_model,
)

def print_rank0(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)

class PolicyTestBase(ABC):
    @staticmethod
    @abstractmethod
    def loss_fn(x: ModelOutput) -> torch.Tensor: ...

    @abstractmethod
    def data_gen_fn(self) -> dict: ...

    @abstractmethod
    def model_fn(self) -> PreTrainedModel: ...

    @abstractmethod
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
    ): ...

    model_class: PreTrainedModel
    config: PretrainedConfig
    microbatch_size: int = 1
    # num_microbatches: int = 4
    num_microbatches: int = 1

    @property
    def world_size(self) -> int:
        return 16
    
    def __init__(self, rank: int, world_size: int):
        import os
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        self.rank = rank
        # self.world_size = world_size
        self.reset_seed()

    def reset_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

class ColossalaiHybridParallelBase(PolicyTestBase):
    def model_fn(self) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        return self.model_class(config)

    def run_hybrid_parallel(
        self,
        tp_size: int,
        pp_size: int,
        sp_size: int,
        sp_mode: str | None,
        fa: bool,
        precision: str,
    ):
        assert precision in ["bf16", "fp32", "fp16"]
        # if fa and precision == "fp32":
        #     raise unittest.SkipTest("Flash Attention does not support fp32")

        test_config = dict(
            # dp_size=1,
            tp_size=tp_size,
            pp_size=pp_size,
            sp_size=sp_size,
            zero_stage=0,
            num_microbatches=self.num_microbatches,
            microbatch_size=self.microbatch_size,
            initial_scale=1,
            enable_flash_attention=fa,
            precision=precision,
        )
        if sp_mode is not None:
            test_config.update(
                {
                    "enable_sequence_parallelism": True,
                    "sequence_parallelism_mode": sp_mode,
                    "sp_size": sp_size,
                }
            )

        (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        ) = self.build_model_from_hybrid_plugin(test_config=test_config)

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

        org_loss, org_output, sharded_loss, sharded_output = (
            self.run_forward_backward_with_hybrid_plugin(
                org_model=org_model,
                sharded_model=sharded_model,
                sharded_optimizer=sharded_optimizer,
                criterion=criterion,
                output_transform_fn=lambda x: x,
                booster=booster,
            )
        )

        org_loss = org_loss.to(
            dtype=torch.bfloat16 if precision == "bf16" else torch.float32 if precision == "fp32" else torch.float16
        )
        if sharded_loss is not None:
            sharded_loss = sharded_loss.to(
                dtype=torch.bfloat16 if precision == "bf16" else torch.float32 if precision == "fp32" else torch.float16
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

    def build_model_from_hybrid_plugin(self, test_config: dict[str, Any]) -> tuple[
        PreTrainedModel,
        Optimizer,
        HybridParallelModule,
        OptimizerWrapper,
        Callable,
        Booster,
    ]:
        use_lazy_init = test_config.pop("use_lazy_init", False)
        precision = test_config.pop("precision")
        precision = torch.bfloat16 if precision == "bf16" else torch.float32 if precision == "fp32" else torch.float16

        ctx = LazyInitContext() if use_lazy_init else nullcontext()
        with ctx:
            org_model = self.model_fn().to(dtype=precision, device="cuda")
            sharded_model = copy.deepcopy(org_model)
        if use_lazy_init:
            ctx.materialize(org_model)

        org_optimizer = Adam(org_model.parameters(), lr=1e-3)
        sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)

        # with patch(
        #     "colossalai.shardformer.shard.sharder.get_autopolicy",
        #     return_value=get_autopolicy(_fullname(org_model)),
        # ):
        policy = get_autopolicy(_fullname(org_model))
        # print_rank0(policy)
        plugin = HybridParallelPlugin(**test_config, custom_policy=policy)
        plugin.precision = None
        booster = Booster(plugin=plugin)

        sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(
            sharded_model, sharded_optimizer, self.loss_fn
        )

        # print_rank0(sharded_model)

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
    ):
        def _criterion(outputs: BaseModelOutputWithPast, inputs: Any):
            outputs = output_transform_fn(outputs)
            loss = criterion(outputs)
            return loss

        data = self.data_gen_fn()
        # print_rank0(f"attn mask in data: {data['attention_mask']}")

        shard_test_data = {}
        for k, v in data.items():
            shard_test_data[k] = v.clone().to("cuda")
        unshard_test_data = {}
        for k, v in data.items():
            unshard_test_data[k] = v.clone().to("cuda")

        org_model.train()
        org_output = org_model(**unshard_test_data)
        org_loss = criterion(org_output)
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


class LlamaPolicyTestClassBase(ColossalaiHybridParallelBase):
    model_class: LlamaPreTrainedModel
    config = LlamaConfig(
        hidden_size=512,
        # hidden_size=128,
        intermediate_size=64,
        # num_attention_heads=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        use_cache=False,
        _attn_implementation="eager",
        # _attn_implementation="flash_attention_2",
    )
    # config = LlamaConfig(max_position_embeddings=4096, use_cache=False, _attn_implementation="eager", num_hidden_layers=4)

    def data_gen_fn(self) -> dict:
        num_batch = self.num_microbatches * self.microbatch_size
        seq_len = 512
        # seq_len = 64
        # attn_mask_rand = torch.randint(1, 2, (num_batch, seq_len)) # B, L
        # attn_mask_rand = torch.randint(1, 2, (num_batch, seq_len, seq_len)) # B, L, L
        attn_mask_rand = torch.randint(0, 2, (num_batch, seq_len, seq_len)) # B, L, L
        # attn_mask_rand = torch.randint(1, 2, (num_batch, seq_len, seq_len)) # B, L, L
        # attn_mask_rand[0, 0, 0] = 0
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, seq_len)),
            "attention_mask": attn_mask_rand,
            # NOTE(runyu): this is for testing anymask, and you could change the internal hugginface transformer code to make it work, like:
            # if attention_mask.bool().all():
            #     causal_mask = self._update_causal_mask(
            #         attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            #     )
            # else:
            #     import numpy as np
            #     assert attention_mask.shape == (inputs_embeds.shape[0], inputs_embeds.shape[1], inputs_embeds.shape[1])
            #     causal_mask = attention_mask.unsqueeze(dim=1).expand(-1, self.config.num_attention_heads, -1, -1)
            #     # set zero part to -inf
            #     causal_mask = torch.where(causal_mask == 0, torch.tensor(-np.inf), causal_mask)
        }

        return input

    def check_fn(
        self,
        booster: Booster,
        org_model: LlamaPreTrainedModel,
        sharded_model: ModelWrapper,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: ModelOutput,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ):
        stage_manager = booster.plugin.stage_manager
        tp_group = booster.plugin.tp_group
        precision = booster.plugin.precision

        # unwrap model
        model = unwrap_model(org_model, "LlamaModel", "model")
        shard_model = unwrap_model(sharded_model, "LlamaModel", "model")

        row_layer_for_check = ["layers[0].self_attn.q_proj", "embed_tokens"]
        col_layer_for_check = ["layers[0].self_attn.o_proj"]

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if stage_manager is None or stage_manager.is_first_stage():
            # atol, rtol = (1e-6, 1e-4) if precision == "fp32" else (5e-3, 5e-3)
            atol, rtol = (7e-3, 7e-3) if precision == "fp32" else (7e-3, 7e-3)
            row_layer_grads = get_grad_tensors_for_check(
                model,
                shard_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                model,
                shard_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            grads_to_check.update(col_layer_grads)
            grads_to_check.update(row_layer_grads)

        # optimizer executes step
        org_optim.step()
        sharded_optim.step()

        # check last hidden state & loss
        if stage_manager is None or stage_manager.is_last_stage():
            atol, rtol = (1e-5, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
            print_rank0(f"org_loss: {org_loss}, sharded_loss: {sharded_loss}")
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
            atol, rtol = (1e-4, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
            # embed_tokens have different dimension, so skip to check row_layer weight
            check_weight(
                model,
                shard_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            print_rank0("weights check passed")

        # check grads
        check_all_grad_tensors(grads_to_check)

class TestLlamaForCausalLMPolicy(LlamaPolicyTestClassBase):

    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)

    @staticmethod
    def loss_fn(x: CausalLMOutputWithPast) -> torch.Tensor:
        return x.loss

    model_class = LlamaForCausalLM

    def data_gen_fn(self) -> dict:
        input = super().data_gen_fn()
        input["labels"] = input["input_ids"]
        return input

    def test_context_parallel(self, tp_size: int, pp_size: int, sp_size: int, sp_mode: str):
        self.run_hybrid_parallel(tp_size, pp_size, sp_size, sp_mode, True, "bf16")
        # self.run_hybrid_parallel(tp_size, pp_size, sp_size, sp_mode, True, "fp16")

def run(rank: int, world_size: int, tp_size: int, pp_size: int, sp_size: int, sp_mode: str):
    test_class = TestLlamaForCausalLMPolicy(rank, world_size)
    print(f"rank {rank} done, tp_size={tp_size}, pp_size={pp_size}, sp_size={sp_size}, sp_mode={sp_mode}")
    test_class.test_context_parallel(tp_size, pp_size, sp_size, sp_mode)

'''
CUDA_VISIBLE_DEVICES=4,5 python -m tests.tmp_test.tmp_test_anymask_llama --world_size 2 --tp_size 2 --pp_size 1 --sp_size 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m tests.tmp_test.tmp_test_anymask_llama --world_size 4 --tp_size 2 --pp_size 2 --sp_size 1
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m tests.tmp_test.tmp_test_anymask_llama --world_size 4 --tp_size 2 --pp_size 1 --sp_size 2 --sp_mode ring_attn
CUDA_VISIBLE_DEVICES=4,5 python -m tests.tmp_test.tmp_test_anymask_llama --world_size 2 --tp_size 1 --pp_size 1 --sp_size 2 --sp_mode ring_attn
CUDA_VISIBLE_DEVICES=0,1 python -m tests.tmp_test.tmp_test_anymask_llama --world_size 2 --tp_size 1 --pp_size 1 --sp_size 2 --sp_mode ring_attn
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m tests.tmp_test.tmp_test_anymask_llama --world_size 4 --tp_size 1 --pp_size 1 --sp_size 4 --sp_mode ring_attn
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m tests.tmp_test.tmp_test_anymask_llama --world_size 4 --tp_size 1 --pp_size 1 --sp_size 4 --sp_mode ring_attn_zig_zag
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m tests.tmp_test.tmp_test_anymask_llama --world_size 4 --tp_size 1 --pp_size 1 --sp_size 4 --sp_mode ring_attn
'''

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--sp_size", type=int, default=4)
    parser.add_argument("--sp_mode", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    from torch.multiprocessing import spawn
    args = parse_args()
    world_size = args.world_size
    tp_size = args.tp_size
    pp_size = args.pp_size
    sp_size = args.sp_size
    sp_mode = args.sp_mode
    assert world_size == tp_size * pp_size * sp_size
    spawn(run, nprocs=world_size, args=(world_size, tp_size, pp_size, sp_size, sp_mode))