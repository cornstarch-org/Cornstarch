from typing import Any, Dict, List, cast

import torch.distributed as dist
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.shardformer.policies.auto_policy import _fullname
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    Policy,
    SubModuleReplacementDescription,
)
from torch import nn

from cornstarch.models.multimodal_language_model import (
    ModalModule,
    ModalModuleType,
    MultimodalProjector,
    MultimodalProjectorConfig,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)
from cornstarch.shardformer.shard.shard_config import ShardConfig


class MultimodalProjectorPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: MultimodalProjectorConfig) -> list[str]:
        assert isinstance(
            config, MultimodalProjectorConfig
        ), f"config must be MultimodalProjectorConfig, got {type(config)}"

        if config.projection_type == "linear":
            return ["projection"]
        elif config.projection_type == "mlp":
            return ["in_proj", "activation", "out_proj"]
        elif config.projection_type == "qformer":
            raise NotImplementedError("QFormer is not supported yet.")

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        pass

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from cornstarch.models.multimodal_language_model import MultimodalProjector

        policy: dict[str | nn.Module, ModulePolicyDescription] = {}

        # if self.shard_config.enable_tensor_parallelism:
        #     # TODO: check if input is in parallel
        #     policy[MultimodalProjector] = ModulePolicyDescription(
        #         sub_module_replacement=[
        #             SubModuleReplacementDescription(
        #                 "projection",
        #                 target_module=Linear1D_Row,
        #                 ignore_if_not_exist=True,
        #             ),
        #             SubModuleReplacementDescription(
        #                 "in_proj",
        #                 target_module=Linear1D_Col,
        #                 ignore_if_not_exist=True,
        #             ),
        #             SubModuleReplacementDescription(
        #                 "out_proj",
        #                 target_module=Linear1D_Row,
        #                 ignore_if_not_exist=True,
        #             ),
        #             # add qformer layers
        #         ]
        #     )

        return policy

    def postprocess(self) -> nn.Module:
        return self.model

    def get_held_layers(self) -> list[nn.Module]:
        assert self.pipeline_stage_manager is not None

        stage_manager: MultiModalPipelineStageManager = self.pipeline_stage_manager
        config = cast(MultimodalProjectorConfig, self.model.config)
        model = cast(MultimodalProjector, self.model)
        held_layers = []

        if config.projection_type == "linear":
            held_layers.append(model.projection)
        elif config.projection_type == "mlp":
            held_layers.extend([model.in_proj, model.activation, model.out_proj])
        elif config.projection_type == "qformer":
            raise NotImplementedError("QFormer is not supported yet.")

        if stage_manager.is_last_stage(check_only_in_modal=True):
            return held_layers

        return []


class ModalModulePolicyMixin:
    pipeline_stage_manager: MultiModalPipelineStageManager

    def should_hold_module(self, modal: PipelineTemplate) -> bool:
        assert self.pipeline_stage_manager is not None

        stage_manager: MultiModalPipelineStageManager = self.pipeline_stage_manager
        assert isinstance(stage_manager, MultiModalPipelineStageManager)
        return dist.get_rank() in stage_manager.pg_mesh.modal_to_ranks[modal]


class ModalModulePolicy(Policy, ModalModulePolicyMixin):
    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from cornstarch.shardformer.policies.auto_policy import get_autopolicy

        model = cast(ModalModule, self.model)
        policies = {}
        if model.modal_type == ModalModuleType.Encoder:
            # Module first
            policy = get_autopolicy(_fullname(model.module))
            policy.set_model(model.module)
            policy.set_shard_config(self.shard_config)
            policies.update(policy.module_policy())

            policy = MultimodalProjectorPolicy()
            policy.set_model(model.projector)
            policy.set_shard_config(self.shard_config)
            policies.update(policy.module_policy())
        else:
            # Projector first
            policy = MultimodalProjectorPolicy()
            policy.set_model(model.projector)
            policy.set_shard_config(self.shard_config)
            policies.update(policy.module_policy())

            policy = get_autopolicy(_fullname(model.module))
            policy.set_model(model.module)
            policy.set_shard_config(self.shard_config)
            policies.update(policy.module_policy())

        return policies

    def postprocess(self) -> nn.Module:
        return self.model

    def get_held_layers(self) -> list[nn.Module]:
        from cornstarch.shardformer.policies.auto_policy import get_autopolicy

        assert self.pipeline_stage_manager is not None

        model = cast(ModalModule, self.model)
        stage_manager: MultiModalPipelineStageManager = self.pipeline_stage_manager
        shard_config: ShardConfig = self.shard_config

        if not self.should_hold_module(shard_config.pipeline_template):
            return []

        held_layers = []
        policy = get_autopolicy(_fullname(model.module))
        policy.set_model(model.module)
        policy.set_shard_config(self.shard_config)
        held_layers.extend(policy.get_held_layers())

        if stage_manager.is_last_stage(check_only_in_modal=True):
            policy = MultimodalProjectorPolicy()
            policy.set_model(model.projector)
            policy.set_shard_config(self.shard_config)
            held_layers.extend(policy.get_held_layers())

        return held_layers


class LanguageModelPolicyWrapper(Policy, ModalModulePolicyMixin):
    """A policy wrapper for language model.

    This is to inject different stage management in multimodal.
    """

    def __init__(self, policy: Policy):
        self.policy = policy

    def set_model(self, model: nn.Module):
        self.policy.set_model(model)

    def set_shard_config(self, shard_config: ShardConfig):
        self.policy.set_shard_config(shard_config)

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.policy.preprocess()

    def postprocess(self) -> nn.Module:
        return self.policy.postprocess()

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        return self.policy.module_policy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def get_held_layers(self) -> list[nn.Module]:
        shard_config: ShardConfig = self.policy.shard_config
        if not self.should_hold_module(shard_config.pipeline_template):
            return []

        return self.policy.get_held_layers()
