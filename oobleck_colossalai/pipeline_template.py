class PipelineTemplate:
    """A template for a single pipeline that can be used to instantiate identical pipelines."""

    def __init__(
        self,
        modules_per_stage: list[list[str]],
        latency: float = 0.0,
        mem_required: int = 0,
    ):
        self.modules_per_stage = modules_per_stage
        self.latency = latency
        self.mem_required = mem_required

    @property
    def num_layers(self) -> int:
        return sum(len(stage) for stage in self.modules_per_stage)

    @property
    def num_stages(self) -> int:
        return len(self.modules_per_stage)
