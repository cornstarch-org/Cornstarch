import copy
from abc import ABC, abstractmethod
from unittest.mock import patch

import torch
from conftest import PolicyTestBase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
)

from cornstarch.shardformer.policies.mixtral import (
    MixtralForCausalLMPolicy,
    MixtralModelPolicy,
)

from ._utils import (
    build_model_from_hybrid_plugin,
    check_all_grad_tensors,
    check_loss,
    check_output_hidden_state,
    check_weight,
    get_grad_tensors_for_check,
    run_forward_backward_with_hybrid_plugin,
    unwrap_model,
)


class MixtralPolicyTestCaseBase(PolicyTestBase, ABC):
    # Implementation for data_gen_fn and loss_fn
    # Copied from colossalai/tests/kit/model_zoo/transformers/mistral.py
    @staticmethod
    @abstractmethod
    def data_gen_fn() -> dict: ...

    @staticmethod
    @abstractmethod
    def loss_fn(x: ModelOutput) -> torch.Tensor: ...

    model_class: MixtralPreTrainedModel
    config = MixtralConfig(
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=64,
        num_hidden_layers=4,
        vocab_size=50258,
        # Need to explicitly set use_cache=False
        # https://github.com/huggingface/transformers/issues/28056
        use_cache=False,
    )

    def model_fn(self) -> MixtralPreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        return self.model_class(config)

    def run_hybrid_parallel(self, tp_size: int, pp_size: int, fa: bool, precision: str):
        # Implementation copied from
        # https://github.com/hpcaitech/ColossalAI/blob/8020f4263095373e4c7ad1b15e54b966a8ccb683/tests/test_shardformer/test_model/test_shard_mistral.py#L26
        test_config = {
            "tp_size": tp_size,
            "pp_size": pp_size,
            "precision": precision,
            "zero_stage": 0,
            "num_microbatches": 4,
            "initial_scale": 1,
            "enable_flash_attention": fa,
        }
        (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        ) = build_model_from_hybrid_plugin(
            model_fn=self.model_fn,
            loss_fn=self.loss_fn,
            test_config=test_config,
        )

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

        org_loss, org_output, sharded_loss, sharded_output = (
            run_forward_backward_with_hybrid_plugin(
                org_model,
                sharded_model,
                sharded_optimizer,
                self.data_gen_fn,
                lambda x: x,  # output_transform_fn,
                criterion,
                booster,
            )
        )

        stage_manager = booster.plugin.stage_manager
        tp_group = booster.plugin.tp_group

        # unwrap model
        mixtral_model = unwrap_model(org_model, "MixtralModel", "model")
        shard_mixtral_model = unwrap_model(sharded_model, "MixtralModel", "model")

        row_layer_for_check = ["layers[0].self_attn.q_proj", "embed_tokens"]
        col_layer_for_check = ["layers[0].self_attn.o_proj"]

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if (
            stage_manager is None or stage_manager.is_first_stage()
        ) and booster.plugin.zero_stage == 0:
            atol, rtol = (5e-5, 1e-4) if precision == "fp32" else (5e-3, 5e-3)
            row_layer_grads = get_grad_tensors_for_check(
                mixtral_model,
                shard_mixtral_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                mixtral_model,
                shard_mixtral_model,
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
        org_optimizer.step()
        sharded_optimizer.step()

        # check last hidden state & loss
        if stage_manager is None or stage_manager.is_last_stage():
            atol, rtol = (1e-5, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
            if org_model.__class__.__name__ == "MixtralModel":
                check_output_hidden_state(
                    org_output, sharded_output, stage_manager, atol=atol, rtol=rtol
                )

            check_loss(org_loss, sharded_loss, atol=1e-5, rtol=1e-3)

        # check weights
        if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
            atol, rtol = (2e-4, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
            # embed_tokens have different dimension, so skip to check row_layer weight
            check_weight(
                mixtral_model,
                shard_mixtral_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )

        # check grads
        check_all_grad_tensors(grads_to_check)


@instantiate_parametrized_tests
class TestMixtralModelPolicy(MixtralPolicyTestCaseBase):
    @staticmethod
    def data_gen_fn() -> dict:
        # Generated from following code snippet
        #
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        # input = 'My favourite condiment is vinegar' (last two words repeated to satisfy length requirement)
        # tokenized_input = tokenizer([input], return_tensors="pt")
        # input_ids = tokenized_input['input_ids']
        # attention_mask = tokenized_input['attention_mask']
        input_ids = torch.tensor(
            [[1, 1984, 16020, 2076, 2487, 349, 21375, 4749]], dtype=torch.int64
        )
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
        return dict(input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def loss_fn(x: BaseModelOutputWithPast) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    model_class = MixtralModel

    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["fp16", "fp32"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=MixtralModelPolicy(),
        ):
            self.run_hybrid_parallel(tp_size, pp_size, fa, precision)


@instantiate_parametrized_tests
class TestMixtralForCausalLMPolicy(MixtralPolicyTestCaseBase):
    @staticmethod
    def data_gen_fn() -> dict:
        data = TestMixtralModelPolicy.data_gen_fn()
        data["labels"] = data["input_ids"].clone()
        return data

    @staticmethod
    def loss_fn(x: CausalLMOutputWithPast) -> torch.Tensor:
        return x.loss

    model_class = MixtralForCausalLM

    @parametrize("tp_size, pp_size", [(2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["fp16", "fp32"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=MixtralForCausalLMPolicy(),
        ):
            self.run_hybrid_parallel(tp_size, pp_size, fa, precision)
