import re
from abc import ABC, abstractmethod

import numpy as np
from cornstarch.pipeline_template import PipelineTemplate
from transformers import PretrainedConfig


class PipelineTemplatePolicyBase(ABC):
    """A policy base that defines the interface for a pipeline template policy."""

    skip_replaced_modules: bool = True

    @staticmethod
    @abstractmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        """Get all modules in the model to create a pipeline template."""
        ...

    @abstractmethod
    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        """Pipeline template sanity check.

        Its implementation should check if the pipeline template is valid for the model.
        Specifically,
        1. check if this pipeline template is for the model that the policy is designed for.
        2. check all modules returned by `get_all_modules` are used in the pipeline template
        3. check whether modules per stage are correctly distributed according to the policy

        Args:
            template (PipelineTemplate): the pipeline template to be checked

        Raises:
            ValueError: if the pipeline template is invalid
        """
        ...

    def set_pipeline_template(self, pipeline_template: PipelineTemplate):
        self.pipeline_template = pipeline_template

    def distribute_layers(
        self, num_layers: int, num_stages: int, is_decoder: bool = True
    ) -> list[int]:
        """Distribute layers to stages."""
        assert hasattr(
            self, "pipeline_template"
        ), "pipeline_template should be set before calling distribute_layers"

        assert num_stages == self.pipeline_template.num_stages, (
            f"num_stages {num_stages} should be equal to "
            f"pipeline_template.num_stages {self.pipeline_template.num_stages}"
        )

        return [
            sum(bool(re.search(r"\.\d", s)) for s in modules)
            for modules in self.pipeline_template.modules_per_stage
        ]

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: int,
        num_model_chunks: int = 1,
        num_stages: int = 0,
    ) -> tuple[int, int]:
        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)

        start_idx = num_layers_per_stage_accumulated[stage]
        end_idx = num_layers_per_stage_accumulated[stage + 1]

        return (start_idx, end_idx)
