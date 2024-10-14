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

from ._utils import (
    check_all_grad_tensors,
    check_loss,
    check_weight,
    get_grad_tensors_for_check,
    unwrap_model,
)


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
    num_microbatches: int = 4

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
        self.rank = rank
        # self.world_size = world_size
        self.reset_seed()



    # def setUp(self) -> None:
    #     super().setUp()
    #     with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
    #         self._spawn_processes()
# 
    # def tearDown(self) -> None:
        # return super().tearDown()
# 
    # @classmethod
    # def _run(cls, rank: int, test_name: str, file_name: str, pipe) -> None:
        # Copy from: torch/testing/_internal/common_fsdp.py FSDPTest._run
        # self = cls(test_name)
        # self.rank = rank
        # self.file_name = file_name
# 
        # print(f"dist init r={self.rank}, world={self.world_size}")
# 
        # backend = "gloo"
# 
        # try:
        #     dist.init_process_group(
        #         init_method=f"{FILE_SCHEMA}{self.file_name}",
        #         backend=backend,
        #         world_size=int(self.world_size),
        #         rank=self.rank,
        #     )
        # except RuntimeError as e:
        #     if "recompile" in e.args[0]:
        #         sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)
# 
            # raise
# 
        # if torch.cuda.is_available() and torch.cuda.device_count():
            # device_id = self.rank % torch.cuda.device_count()
            # torch.cuda.set_device(device_id)
# 
        # self.reset_seed()
# 
        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        # dist.barrier()
# 
        # with torch.backends.cudnn.flags(
            # enabled=True, deterministic=True, benchmark=True
        # ):
            # self.run_test(test_name, pipe)
# 
        # dist.barrier()
# 
        # dist.destroy_process_group()
# 
    def reset_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
# 
    # @staticmethod
    # def batch_isend_irecv_gloo(p2p_op_list: list[dist.P2POp]) -> list[dist.Work]:
        # reqs: list[tuple[dist.Work, torch.Tensor]] = []
        # for p2p_op in p2p_op_list:
            # if p2p_op.op == dist.isend:
                # tensor = p2p_op.tensor.to("cpu")
                # work = p2p_op.op(tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
            # else:
                # tensor = torch.empty_like(p2p_op.tensor, device="cpu")
                # work = p2p_op.op(tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
# 
            # reqs.append((work, tensor))
# 
        # send_reqs = []
        # with torch.no_grad():
            # for (req, tensor), p2p_op in zip(reqs, p2p_op_list):
                # if req is None:
                    # continue
# 
                # if p2p_op.op == dist.irecv:
                    # req.wait()
                    # p2p_op.tensor.copy_(tensor)
                # else:
                    # send_reqs.append(req)
# 
        # return send_reqs
# 
    # @staticmethod
    # def all_to_all_gloo(
        # output_tensor_list: list[torch.Tensor],
        # input_tensor_list: list[torch.Tensor],
        # group: Optional[dist.ProcessGroup] = None,
        # async_op: Optional[bool] = False,
    # ):
        # """Backend gloo doesn't support all_to_all, so we simulate it here."""
        # world_size = dist.get_world_size(group)
        # rank = dist.get_rank(group)
# 
        # Each rank gathers the "i-th" input from all ranks
        # for i in range(world_size):
            # chunk = input_tensor_list[i].to("cpu")  # Each rank's i-th input tensor
            # gathered_tensors = [
                # torch.empty_like(chunk) for _ in range(world_size)
            # ]  # Buffers for allgather
# 
            # Perform all_gather to collect the i-th tensor from all ranks
            # dist.all_gather(gathered_tensors, chunk, group)
# 
            # if i == rank:
                # assert len(output_tensor_list) == len(gathered_tensors)
                # for output_tensor, gathered_tensor in zip(
                    # output_tensor_list, gathered_tensors
                # ):
                    # output_tensor.copy_(gathered_tensor)
# 
    # @staticmethod
    # def all_to_all_single_gloo(
        # output: torch.Tensor,
        # input: torch.Tensor,
        # group: Optional[dist.ProcessGroup] = None,
        # async_op: Optional[bool] = False,
    # ):
        # world_size = dist.get_world_size(group)
        # rank = dist.get_rank(group)
# 
        # chunk_size = input.size(0) // world_size
        # gathered_tensors = [torch.empty_like(input) for _ in range(world_size)]
# 
        # dist.all_gather(gathered_tensors, input, group)
# 
        # output_tensor = torch.cat(
            # [
                # gathered_tensors[i][rank * chunk_size : (rank + 1) * chunk_size]
                # for i in range(world_size)
            # ],
            # dim=0,
        # )
        # output.copy_(output_tensor)


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
        assert precision in ["bf16", "fp32"]
        # if fa and precision == "fp32":
        #     raise unittest.SkipTest("Flash Attention does not support fp32")

        test_config = dict(
            tp_size=tp_size,
            pp_size=pp_size,
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

        # with (
        #     patch.object(
        #         dist,
        #         "batch_isend_irecv",
        #         new=PolicyTestBase.batch_isend_irecv_gloo,
        #     ),
        #     patch.object(
        #         dist,
        #         "all_to_all",
        #         new=PolicyTestBase.all_to_all_gloo,
        #     ),
        #     patch.object(
        #         dist,
        #         "all_to_all_single",
        #         new=PolicyTestBase.all_to_all_single_gloo,
        #     ),
        #     patch(
        #         "colossalai.pipeline.p2p._check_device",
        #         return_value=(torch.device("cuda"), False),
        #     ),
        # ):
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
            dtype=torch.bfloat16 if precision == "bf16" else torch.float32
        )
        if sharded_loss is not None:
            sharded_loss = sharded_loss.to(
                dtype=torch.bfloat16 if precision == "bf16" else torch.float32
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
        precision = torch.bfloat16 if precision == "bf16" else torch.float32

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
        plugin = HybridParallelPlugin(**test_config)
        plugin.precision = None
        booster = Booster(plugin=plugin)

        sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(
            sharded_model, sharded_optimizer, self.loss_fn
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
    ):
        def _criterion(outputs: BaseModelOutputWithPast, inputs: Any):
            outputs = output_transform_fn(outputs)
            loss = criterion(outputs)
            return loss

        data = self.data_gen_fn()

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
        hidden_size=128,
        intermediate_size=64,
        num_attention_heads=16,
        num_hidden_layers=4,
        use_cache=False,
        _attn_implementation="eager",
    )

    def data_gen_fn(self) -> dict:
        num_batch = self.num_microbatches * self.microbatch_size
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 64)),
            "attention_mask": torch.ones(num_batch, 64),
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
            atol, rtol = (1e-6, 1e-4) if precision == "fp32" else (5e-3, 5e-3)
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

def run(rank: int, world_size: int):
    test_class = TestLlamaForCausalLMPolicy(rank, world_size)
    test_class.test_context_parallel(1, 1, world_size, "ring_attn")

if __name__ == "__main__":
    from torch.multiprocessing import spawn
    world_size = 4
    spawn(run, nprocs=world_size, args=(world_size,))