import copy
import os
import random
import sys
from abc import ABC, abstractmethod
from typing import Optional
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer._utils import getattr_
from colossalai.shardformer.layer.qkv_fused_linear import (
    FusedLinear1D_Col,
    gather_fused_qkv_in_gpt2_style,
)
from colossalai.tensor.d_tensor.api import (
    is_customized_distributed_tensor,
    is_distributed_tensor,
)
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.testing import assert_close
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel


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
