from torch import nn

from cornstarch.pipeline_template import PipelineTemplate


class Module(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(num_layers)])
        self.start_idx, self.end_idx = 0, num_layers

    def forward(self, x):
        for layer in self.layer[self.start_idx : self.end_idx]:
            x = layer(x)
        return x


encoder1_template = PipelineTemplate(
    "encoder1", [["layer.0", "layer.1"], ["layer.2", "layer.3"]]
)
encoder2_template = PipelineTemplate(
    "encoder2", [["layer.0", "layer.1"], ["layer.2", "layer.3"], ["layer.4", "layer.5"]]
)
encoder3_template = PipelineTemplate(
    "encoder3", [["layer.0", "layer.1", "layer.2"], ["layer.3", "layer.4"]]
)

llm_template_2stages = PipelineTemplate(
    "llm", [["layer.0", "layer.1", "layer.2"], ["layer.3", "layer.4"]]
)
llm_template_4stages = PipelineTemplate(
    "llm",
    [
        ["layer.0", "layer.1", "layer.2"],
        ["layer.3"],
        ["layer.4", "layer.5", "layer.6", "layer.7"],
        ["layer.8", "layer.9"],
    ],
)
