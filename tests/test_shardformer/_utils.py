import copy
import os
import random
import sys
from contextlib import nullcontext
from typing import Any, Callable, Optional
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.lazy import LazyInitContext
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer._utils import getattr_
from colossalai.tensor.d_tensor.api import (
    is_customized_distributed_tensor,
    is_distributed_tensor,
)
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.testing import assert_close
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA


class PolicyTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def setUp(self) -> None:
        super().setUp()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self._spawn_processes()

    def tearDown(self) -> None:
        return super().tearDown()

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, pipe) -> None:
        # Copy from: torch/testing/_internal/common_fsdp.py FSDPTest._run
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        backend = "nccl" if torch.cuda.is_available() else "gloo"

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

        device_ids = None
        if torch.cuda.is_available() and torch.cuda.device_count():
            device_id = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            device_ids = [device_id]

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier(device_ids=device_ids)

        with torch.backends.cudnn.flags(
            enabled=True, deterministic=True, benchmark=True
        ):
            self.run_test(test_name, pipe)

        dist.barrier(device_ids=device_ids)

        dist.destroy_process_group()


def build_model_from_hybrid_plugin(
    model_fn: Callable, loss_fn: Callable, test_config: dict[str, Any]
):
    use_lazy_init = False
    if "use_lazy_init" in test_config:
        use_lazy_init = test_config.pop("use_lazy_init")

    ctx = LazyInitContext() if use_lazy_init else nullcontext()
    with ctx:
        org_model = model_fn()
        sharded_model = copy.deepcopy(org_model)
    if use_lazy_init:
        ctx.materialize(org_model)
    org_model = org_model.cuda()
    org_optimizer = Adam(org_model.parameters(), lr=1e-3)
    sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)
    criterion = loss_fn

    plugin = HybridParallelPlugin(**test_config)
    booster = Booster(plugin=plugin)

    sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(
        sharded_model, sharded_optimizer, criterion
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
    org_model: nn.Module,
    sharded_model: nn.Module,
    sharded_optimizer: Optimizer,
    data_gen_fn: Callable,
    output_transform_fn: Callable,
    criterion: Callable,
    booster: Booster,
):
    org_model.cuda()
    sharded_model.cuda()

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    data = data_gen_fn()

    shard_test_data = {}
    for k, v in data.items():
        shard_test_data[k] = data[k].clone()
    unshard_test_data = {}
    for k, v in data.items():
        unshard_test_data[k] = data[k].clone()

    sharded_model.train()
    if booster.plugin.stage_manager is not None:
        for k, v in shard_test_data.items():
            if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
                new_shape = [1] * v.dim()
                new_shape[0] = 4
                shard_test_data[k] = v.to("cuda").repeat(*new_shape)

        data_iter = iter([shard_test_data])
        sharded_output = booster.execute_pipeline(
            data_iter,
            sharded_model,
            _criterion,
            sharded_optimizer,
            return_loss=True,
            return_outputs=True,
        )
        sharded_loss = sharded_output["loss"]

    else:
        shard_test_data = {k: v.cuda() for k, v in shard_test_data.items()}
        sharded_output = sharded_model(**shard_test_data)
        sharded_loss = criterion(sharded_output)
        sharded_optimizer.backward(sharded_loss)

    org_model.train()
    if booster.plugin.stage_manager is not None:
        for k, v in unshard_test_data.items():
            if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
                new_shape = [1] * v.dim()
                new_shape[0] = 4
                unshard_test_data[k] = v.to("cuda").repeat(*new_shape)
    unshard_test_data = {k: v.cuda() for k, v in unshard_test_data.items()}
    org_output = org_model(**unshard_test_data)
    org_loss = criterion(org_output)
    org_loss.backward()

    return org_loss, org_output, sharded_loss, sharded_output


def check_output_hidden_state(
    org_output: Tensor,
    sharded_output: Tensor,
    stage_manager: Optional[PipelineStageManager] = None,
    atol: float = 1e-5,
    rtol: float = 1e-3,
):
    org_hidden_state = org_output.last_hidden_state

    if stage_manager and stage_manager.is_last_stage(ignore_chunk=True):
        sharded_hidden_state = sharded_output["outputs"]["last_hidden_state"]
    else:
        sharded_hidden_state = sharded_output.last_hidden_state

    assert_close(
        org_hidden_state.float(), sharded_hidden_state.float(), atol=atol, rtol=rtol
    )


def check_loss(
    org_loss: Tensor, sharded_loss: Tensor, atol: float = 1e-5, rtol: float = 1e-3
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
            sharded_weight_list = [
                torch.zeros_like(sharded_weight).to("cuda")
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
            shard_grad_list = [
                torch.zeros_like(shard_grad).to("cuda")
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
                torch.zeros_like(shard_grad).to("cuda")
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
