from transformers import GPT2ForSequenceClassification

from oobleck_colossalai.pipeline_template import PipelineTemplate


def test_get_model_name(model: GPT2ForSequenceClassification):
    assert (
        PipelineTemplate.get_model_name(model)
        == "transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification"
    )


def test_get_modules(model: GPT2ForSequenceClassification):
    modules = PipelineTemplate.get_modules(model)
    assert modules == [
        "transformer.wte",
        "transformer.wpe",
        "transformer.drop",
        *[f"transformer.h.{i}" for i in range(model.config.num_hidden_layers)],
        "transformer.ln_f",
        "score",
    ]


def test_sanity_check(model: GPT2ForSequenceClassification):
    pass
