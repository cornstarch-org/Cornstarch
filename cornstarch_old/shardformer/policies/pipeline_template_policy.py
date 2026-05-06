from abc import ABC, abstractmethod

from transformers import PretrainedConfig

from cornstarch.pipeline_template import PipelineTemplate


class PipelineTemplatePolicyBase(ABC):
    """A policy base that defines the interface for a pipeline template policy."""

    # skip_replaced_modules: bool = True

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
