"""LoRA remains trainable after Cornstarch tensor-parallel materialization."""

import torch
from peft import LoraConfig

from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import configure_finetuning, from_hf_config
from tests.distributed.distributed_base import GlooDistributedTestBase
from tests.model.model_configs import llama_config


class TestLoRATensorParallel(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_all_linear_lora_backward_with_tensor_parallelism(self) -> None:
        model = from_hf_config(
            llama_config(), model_kind="language", attn_implementation="eager"
        )
        model.set_random_init()
        configure_finetuning(
            model,
            "lora",
            lora_config=LoraConfig(target_modules="all-linear", r=2, lora_alpha=4),
        )

        plan = ParallelizationPlan()
        plan.parallelize(
            model,
            ParallelConfig(tensor_parallel_size=2, data_parallel_size=1),
        )
        plan.materialize("cpu")

        output = model(input_ids=torch.tensor([[1, 2, 3, 4]]))
        output.logits.sum().backward()

        adapter_parameters = [
            parameter for name, parameter in model.named_parameters() if "lora_" in name
        ]
        base_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_" not in name
        ]
        self.assertTrue(adapter_parameters)
        self.assertTrue(
            any(parameter.grad is not None for parameter in adapter_parameters)
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in base_parameters)
        )
