import torch


class PipelineTemplate:
    """A template for a single pipeline that can be used to instantiate identical pipelines.

    TODO (insujang): analyze optimal partitioning of model layers to pipeline stages.
    TODO (insujang): implement to assign different number of layers and different number of
    GPUs to each stage.
    """

    def __init__(
        self,
        node_ids: list[str],
        gpus_per_stage: list[int],
        modules_per_stage: list[list[torch.nn.Module]],
    ):
        self.node_ids = node_ids
        self.gpus_per_stage = gpus_per_stage
        self.modules_per_stage = modules_per_stage
        self.num_layers = sum([len(modules) for modules in modules_per_stage])

    @property
    def num_gpus(self):
        return sum(self.gpus_per_stage)

    @staticmethod
    def create_pipeline_template(
        node_ids: list[str],
        gpus_per_stage: list[int],
        modules: list[torch.nn.Module],
    ):
        """Create a pipeline template.

        Analyzing the given modules, this method creates a pipeline template
        that distributes modules to pipeline stages evenly considering
        their computational costs.
        """
        raise NotImplementedError("This method has not been implemented yet.")
