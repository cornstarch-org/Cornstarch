from __future__ import annotations

import torch
from transformers import PretrainedConfig


class Pipeline:
    """A pipeline class that is instantiated from a pipeline template."""

    def __init__(self, pipeline_template: PipelineTemplate, ranks: list[int]):
        assert len(ranks) == pipeline_template.num_gpus * len(
            pipeline_template.node_ids
        )
        self.pipeline_template = pipeline_template
        self.ranks = ranks


class PipelineTemplate:
    """A template for a single pipeline that can be used to instantiate identical pipelines.

    TODO (insujang): analyze optimal partitioning of model layers to pipeline stages.
    TODO (insujang): implement to assign different number of layers and different number of
    GPUs to each stage.
    """

    def __init__(
        self,
        model_config: PretrainedConfig,
        node_ids: list[str],
        gpus_per_stage: int,
        module_names_per_stage: list[list[str]],
    ):
        self.model_config = model_config
        self.node_ids = node_ids
        self.gpus_per_stage = gpus_per_stage
        self.module_names_per_stage = module_names_per_stage
        self.num_layers = sum([len(modules) for modules in module_names_per_stage])

    @property
    def num_stages(self) -> int:
        return len(self.gpus_per_stage)

    @property
    def num_gpus(self) -> int:
        return sum(self.gpus_per_stage)

    def verify_all_modules_in_stage(
        self, model: torch.nn.Module, stage_index: int
    ) -> bool:
        """Verify that all modules are included in a stage of the pipeline template.

        This is for integrity check of the pipeline template of specific model
        after partitioning.
        """
        all_params = {name for name, _ in model.named_parameters()}

        # Iterate over the module names
        for module_name in self.module_names_per_stage[stage_index]:
            try:
                submodule = model.get_submodule(module_name)
            except AttributeError as e:
                raise ValueError(
                    f"Module {module_name} is not found in the model."
                ) from e

            # Remove the parameters of this submodule from the set
            for name, _ in submodule.named_parameters(recurse=True):
                prefixed_name = f"{module_name}.{name}" if module_name else name
                all_params.discard(prefixed_name)

        # If all_params is not empty, meaning some parameters are not covered by
        # the pipeline template
        if all_params:
            raise ValueError(
                f"Following parameters are not covered by the pipeline template: "
                f"{all_params}."
            )

        return True

    def verify_all_modules_in_template(self, model: torch.nn.Module) -> bool:
        """Verify that all modules are included in the pipeline template.

        This is for integrity check of the pipeline template of specific model.
        """
        all_params = {name for name, _ in model.named_parameters()}

        # Iterate over the module names
        for module_name in [
            module_name
            for modules in self.module_names_per_stage
            for module_name in modules
        ]:
            try:
                submodule = model.get_submodule(module_name)
            except AttributeError as e:
                raise ValueError(
                    f"Module {module_name} is not found in the model."
                ) from e

            # Remove the parameters of this submodule from the set
            for name, _ in submodule.named_parameters(recurse=True):
                prefixed_name = f"{module_name}.{name}" if module_name else name
                all_params.discard(prefixed_name)

        # If all_params is not empty, meaning some parameters are not covered by
        # the pipeline template
        if all_params:
            raise ValueError(
                f"Following parameters are not covered by the pipeline template: "
                f"{all_params}."
            )

        return True

    @staticmethod
    def create_pipeline_template(
        node_ids: list[str],
        gpus_per_stage: list[int],
        module_names: list[str],
    ):
        """Create a pipeline template.

        Analyzing the given modules, this method creates a pipeline template
        that distributes modules to pipeline stages evenly considering
        their computational costs.
        """
        raise NotImplementedError("This method has not been implemented yet.")
