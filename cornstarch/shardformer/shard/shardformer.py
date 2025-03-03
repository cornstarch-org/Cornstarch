from typing import Iterator, List, Optional, Set

from colossalai.accelerator import get_accelerator
from colossalai.shardformer._utils import getattr_, setattr_
from colossalai.shardformer.policies.base_policy import (
    Policy,
    SubModuleReplacementDescription,
)
from colossalai.shardformer.shard.sharder import ModelSharder as ColossalModelSharder
from colossalai.shardformer.shard.shardformer import ShardConfig
from colossalai.shardformer.shard.shardformer import ShardFormer as ColossalShardFormer
from torch import Tensor, nn
from transformers.utils.logging import get_logger

from cornstarch.shardformer.shard.placeholder import TensorPlaceholder

logger = get_logger(__name__)


class ModelSharder(ColossalModelSharder):
    def shard(self) -> list[dict[int, Tensor]]:
        r"""
        Shard the model according to the policy.

        Switch order of _replace_module() and _replace_unheld_layers call
        from the original shard() function implementation
        """
        self.policy.set_model(self.model)
        self.policy.set_shard_config(self.shard_config)
        self._preprocess()
        # get shared params before release unheld layers, this avoid misjudgment of shared params (None is None)
        shared_params = self.policy.get_shared_params()
        self._replace_module()
        self._release_unheld_layers()
        self._materialize()
        self._postprocess()
        return shared_params

    def _materialize(self) -> None:
        for p in self.model.parameters():
            if p.device.type == "meta":
                p.to_empty(device=get_accelerator().get_current_device())
        super()._materialize()

    @classmethod
    def set_tensors_to_placeholder(
        cls, model: nn.Module, exclude: set[nn.Module] = set()
    ) -> None:
        """Set all parameters and buffers of model to TensorPlaceholder instances"""
        if model in exclude:
            return

        for child in model.children():
            cls.set_tensors_to_placeholder(child, exclude=exclude)

        param_holders: dict[str, TensorPlaceholder] = {}
        buffer_holders: dict[str, TensorPlaceholder] = {}
        for n, p in list(model.named_parameters(recurse=False)):
            param_holders[n] = TensorPlaceholder(p)
            setattr(model, n, None)
        for n, buf in list(model.named_buffers(recurse=False)):
            buffer_holders[n] = TensorPlaceholder(buf)
            setattr(model, n, None)

        setattr(model, "_parameter_placeholders", param_holders)
        setattr(model, "_buffer_placeholders", buffer_holders)

    @classmethod
    def _placeholders(
        cls, model: nn.Module, placeholders_name: str
    ) -> Iterator[tuple[nn.Module, str, TensorPlaceholder]]:
        assert placeholders_name in ["_parameter_placeholders", "_buffer_placeholders"]

        for child in model.children():
            yield from cls._placeholders(child, placeholders_name)

        if not hasattr(model, placeholders_name):
            return

        placeholders: dict[str, TensorPlaceholder] = getattr(model, placeholders_name)
        for name, placeholder in placeholders.items():
            yield model, name, placeholder

    @classmethod
    def buffer_placeholders(
        cls,
        model: nn.Module,
        delete_placeholders_after: bool = False,
    ) -> Iterator[tuple[nn.Module, str, TensorPlaceholder]]:
        yield from cls._placeholders(model, "_buffer_placeholders")
        if delete_placeholders_after:
            delattr(model, "_buffer_placeholders")

    @classmethod
    def parameter_placeholders(
        cls,
        model: nn.Module,
        delete_placeholders_after: bool = False,
    ) -> Iterator[tuple[nn.Module, str, TensorPlaceholder]]:
        yield from cls._placeholders(model, "_parameter_placeholders")
        if delete_placeholders_after:
            delattr(model, "_parameter_placeholders")

    def _release_unheld_layers(self) -> Optional[set[nn.Module]]:
        if self.shard_config and self.shard_config.pipeline_stage_manager:
            held_layers = self.policy.get_held_layers()
            self.set_tensors_to_placeholder(self.model, exclude=set(held_layers))
            return set(self._get_recursive_held_layers(held_layers))
        return None

    def _replace_sub_module(
        self,
        org_layer: nn.Module,
        sub_module_replacement: List[SubModuleReplacementDescription],
        include: Optional[Set[nn.Module]] = None,
    ) -> None:
        """
        Override the original method's behavior that raises AssertionError
        when it tries to replace a module that has already been replaced.
        """
        for description in sub_module_replacement:
            suffix = description.suffix
            target_module = description.target_module
            kwargs = {} if description.kwargs is None else description.kwargs

            assert target_module is not None, "target_module should not be None"

            native_sub_module = getattr_(org_layer, suffix, ignore=True)
            if isinstance(native_sub_module, target_module):
                # Skip replacement if submodule has already been replaced
                continue

            # if it is None and we are allowed to ignore this module
            # just skip
            if description.ignore_if_not_exist and native_sub_module is None:
                continue

            try:
                replace_layer = target_module.from_native_module(
                    native_sub_module,
                    process_group=self.shard_config.tensor_parallel_process_group,
                    **kwargs,
                )
            except Exception:
                logger.error(
                    f"Failed to replace {suffix} of type {native_sub_module.__class__.__qualname__}"
                    f" with {target_module.__qualname__}. "
                    "Please check your model configuration or sharding policy, you can set up an issue for us to help you as well."
                )
                raise

            setattr_(org_layer, suffix, replace_layer)

    @staticmethod
    def has_placeholders(model: nn.Module) -> bool:
        return hasattr(model, "_parameter_placeholders")


class ShardFormer(ColossalShardFormer):
    """
    Parallelize model based on the given config and policy.

    In original ColossalAI's ShardFormer, it removes all unheld layers
    by setting their parameters and buffers to None.

    Oobleck training may restore those parameters and buffers, so we
    instead replace them with placeholders. This way, we can restore
    the original model when failures happen and model sharding is changed.
    """

    def __init__(self, shard_config: ShardConfig):
        self.shard_config = shard_config

    def optimize(
        self, model: nn.Module, policy: Policy = None
    ) -> tuple[nn.Module, list[dict[int, Tensor]]]:
        r"""
        This method will optimize the model based on the given policy.

        Args:
            model (`torch.nn.Model`): the origin huggingface model
            shard_config (`ShardConfig`): the config for distribute information
            policy (`Policy`): the custom policy for sharding

        Returns: the sharded model and the shared parameters
        """
        sharder = ModelSharder(
            model=model, shard_config=self.shard_config, policy=policy
        )
        shared_params = sharder.shard()
        return model, shared_params
