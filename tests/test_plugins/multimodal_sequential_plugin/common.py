from cornstarch.pipeline_template import PipelineTemplate

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
    "llm", [["layer.0", "layer.1", "layer.2"], ["layer.3", "layer.4", "layer.5"]]
)
llm_template_4stages = PipelineTemplate(
    "llm",
    [
        ["layer.0", "layer.1"],
        ["layer.2", "layer.3"],
        ["layer.4", "layer.5"],
        ["layer.6", "layer.7"],
    ],
)
