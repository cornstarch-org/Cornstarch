import copy
import os
import random
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, Optional
from unittest.mock import patch

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
from torch.testing import assert_close
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from cornstarch.shardformer.policies.auto_policy import get_autopolicy


class ModelClassBase(ABC):
    rtol, atol = 5e-3, 5e-3
    col_layers_to_check: list[str] = []
    row_layers_to_check: list[str] = []
    norm_layers_to_check: list[str] = []

    def __init__(self, model_class: PreTrainedModel, config: PretrainedConfig):
        self.model_class = model_class
        self.config = config

    @abstractmethod
    def loss_fn(self, x: ModelOutput) -> torch.Tensor: ...

    @abstractmethod
    def data_gen_fn(self, num_batch: int) -> dict: ...

    def model_fn(self, fa: bool) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "flash_attention_2" if fa else "eager"
        return self.model_class(config)


class PolicyTestBase(MultiProcessTestCase, ABC):

    microbatch_size: int = 1
    num_microbatches: int = 4

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
    ):
        """Each ParallelBase class should implement this method"""
        ...

    @property
    def world_size(self) -> int:
        return 16

    def setUp(self) -> None:
        super().setUp()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self._spawn_processes()

    def tearDown(self) -> None:
        return super().tearDown()

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, pipe, **kwargs) -> None:
        # Copy from: torch/testing/_internal/common_fsdp.py FSDPTest._run
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        backend = "gloo"

        try:
            dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}",
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        if torch.cuda.is_available() and torch.cuda.device_count():
            device_id = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)

        self.reset_seed()

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier()

        with torch.backends.cudnn.flags(
            enabled=True, deterministic=True, benchmark=True
        ):
            self.run_test(test_name, pipe)

        dist.barrier()

        dist.destroy_process_group()

    def reset_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)


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


def batch_isend_irecv_gloo(p2p_op_list: list[dist.P2POp]) -> list[dist.Work]:
    reqs: list[tuple[dist.Work, torch.Tensor]] = []
    for p2p_op in p2p_op_list:
        if p2p_op.op == dist.isend:
            tensor = p2p_op.tensor.to("cpu")
            work = p2p_op.op(tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        else:
            tensor = torch.empty_like(p2p_op.tensor, device="cpu")
            work = p2p_op.op(tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)

        reqs.append((work, tensor))

    send_reqs = []
    with torch.no_grad():
        for (req, tensor), p2p_op in zip(reqs, p2p_op_list):
            if req is None:
                continue

            if p2p_op.op == dist.irecv:
                req.wait()
                p2p_op.tensor.copy_(tensor)
            else:
                send_reqs.append(req)

    return send_reqs


def all_to_all_gloo(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
    async_op: Optional[bool] = False,
):
    """Backend gloo doesn't support all_to_all, so we simulate it here."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    # Each rank gathers the "i-th" input from all ranks
    for i in range(world_size):
        chunk = input_tensor_list[i].to("cpu")  # Each rank's i-th input tensor
        gathered_tensors = [
            torch.empty_like(chunk) for _ in range(world_size)
        ]  # Buffers for allgather

        # Perform all_gather to collect the i-th tensor from all ranks
        dist.all_gather(gathered_tensors, chunk, group)

        if i == rank:
            assert len(output_tensor_list) == len(gathered_tensors)
            for output_tensor, gathered_tensor in zip(
                output_tensor_list, gathered_tensors
            ):
                output_tensor.copy_(gathered_tensor)


def all_to_all_single_gloo(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: Optional[bool] = False,
):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    chunk_size = input.size(0) // world_size
    gathered_tensors = [torch.empty_like(input) for _ in range(world_size)]

    dist.all_gather(gathered_tensors, input, group)

    output_tensor = torch.cat(
        [
            gathered_tensors[i][rank * chunk_size : (rank + 1) * chunk_size]
            for i in range(world_size)
        ],
        dim=0,
    )
    output.copy_(output_tensor)


def check_output_hidden_state(
    org_output: Tensor,
    sharded_output: Tensor,
    stage_manager: Optional[PipelineStageManager] = None,
    atol: float = 1e-5,
    rtol: float = 1e-3,
):
    assert (
        stage_manager is None or stage_manager.is_last_stage()
    ), "check_output_hidden_state only supports last stage"
    org_hidden_state = org_output.last_hidden_state
    sharded_hidden_state = sharded_output.last_hidden_state

    assert_close(
        org_hidden_state.float(), sharded_hidden_state.float(), atol=atol, rtol=rtol
    )


def check_loss(
    org_loss: Tensor,
    sharded_loss: Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-3,
):
    assert torch.allclose(org_loss.float(), sharded_loss.float(), atol=atol, rtol=rtol)


def check_weight(
    org_model: nn.Module,
    sharded_model: nn.Module,
    layer_suffix: list[str],
    tp_group: Optional[dist.ProcessGroup] = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
):
    for suffix in layer_suffix:
        org_weight = getattr_(org_model, suffix).weight
        sharded_weight = getattr_(sharded_model, suffix).weight

        # skip if layer is not held by this process
        if sharded_weight is None:
            continue

        if is_distributed_tensor(sharded_weight) or is_customized_distributed_tensor(
            sharded_weight
        ):
            sharded_module = getattr_(sharded_model, suffix)
            if isinstance(sharded_module, FusedLinear1D_Col):
                sharded_weight = gather_fused_qkv_in_gpt2_style(
                    sharded_weight, sharded_module.n_fused, tp_group
                )
            else:
                sharded_weight_list = [
                    torch.zeros_like(sharded_weight, device="cuda")
                    for _ in range(dist.get_world_size(tp_group))
                ]
                dist.all_gather(sharded_weight_list, sharded_weight, tp_group)
                sharded_weight = torch.cat(sharded_weight_list, dim=dim)

        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' weight: {org_weight}, {sharded_weight}")

        assert_close(org_weight.float(), sharded_weight.float(), atol=atol, rtol=rtol)


def get_grad_tensors_for_check(
    org_model: nn.Module,
    sharded_model: nn.Module,
    layer_suffix: list[str],
    tp_group: dist.ProcessGroup = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
    name: str = None,
):
    grad_to_check = {}
    for suffix in layer_suffix:
        org_grad = getattr_(org_model, suffix).weight.grad
        shard_grad = getattr_(sharded_model, suffix).weight.grad
        shard_weight = getattr_(sharded_model, suffix).weight
        if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(
            shard_weight
        ):
            sharded_module = getattr_(sharded_model, suffix)
            if isinstance(sharded_module, FusedLinear1D_Col):
                shard_grad = gather_fused_qkv_in_gpt2_style(
                    shard_grad, sharded_module.n_fused, tp_group
                )
            else:
                shard_grad_list = [
                    torch.zeros_like(shard_grad, device="cuda")
                    for _ in range(dist.get_world_size(tp_group))
                ]
                dist.all_gather(shard_grad_list, shard_grad, tp_group)
                shard_grad = torch.cat(shard_grad_list, dim=dim)

        # embedding may be resized when using tensor parallel
        if shard_grad.shape[0] > org_grad.shape[0]:
            shard_grad = shard_grad[: org_grad.shape[0], :]
        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' grad: {org_grad}, {shard_grad}")

        grad_to_check[suffix] = {
            "org_grad": org_grad.float(),
            "shard_grad": shard_grad.float(),
            "rtol": rtol,
            "atol": atol,
        }

    return grad_to_check


# used by sam/blip2
def check_grad(
    org_model: nn.Module,
    sharded_model: nn.Module,
    layer_suffix: list[str],
    tp_group: dist.ProcessGroup = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
):
    for suffix in layer_suffix:
        org_grad = getattr_(org_model, suffix).weight.grad
        shard_grad = getattr_(sharded_model, suffix).weight.grad
        shard_weight = getattr_(sharded_model, suffix).weight
        if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(
            shard_weight
        ):
            shard_grad_list = [
                torch.zeros_like(shard_grad, device="cuda")
                for _ in range(dist.get_world_size(tp_group))
            ]
            dist.all_gather(shard_grad_list, shard_grad, tp_group)
            shard_grad = torch.cat(shard_grad_list, dim=dim)

        # embedding may be resized when using tensor parallel
        if shard_grad.shape[0] > org_grad.shape[0]:
            shard_grad = shard_grad[: org_grad.shape[0], :]
        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' grad: {org_grad}, {shard_grad}")

        assert_close(org_grad.float(), shard_grad.float(), rtol=rtol, atol=atol)


def unwrap_model(
    module: nn.Module,
    base_model_class_name: Optional[str] = None,
    base_model_attribute_name: Optional[str] = None,
):
    if isinstance(module, HybridParallelModule):
        module = module.unwrap()
    if base_model_class_name is None:
        return module
    if module.__class__.__name__ == base_model_class_name:
        return module
    return getattr(module, base_model_attribute_name, None)


def check_all_grad_tensors(check_tensors):
    """
    "org_grad": tensor to be compared from the original model
    "shard_grad": tensor to be compared from the sharded model
    """
    for suffix, check_info in check_tensors.items():
        org_grad = check_info["org_grad"]
        shard_grad = check_info["shard_grad"]
        rtol = check_info["rtol"]
        atol = check_info["atol"]
        assert_close(org_grad, shard_grad, atol=atol, rtol=rtol)
