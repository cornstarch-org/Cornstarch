from contextlib import nullcontext

import pytest
from accelerate import init_empty_weights
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.gpt2 import (
    GPT2ForSequenceClassificationPolicy,
)
from transformers import GPT2Config, GPT2ForSequenceClassification

num_layers = 4


@pytest.fixture
def model() -> GPT2ForSequenceClassification:
    config = GPT2Config.from_pretrained("gpt2")
    config.is_decoder = True
    config.num_hidden_layers = num_layers
    with init_empty_weights():
        return GPT2ForSequenceClassification(config)


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


@pytest.mark.parametrize(
    "modules_per_stage, expected_check_pass",
    [
        [
            # All layers in a single stage
            [
                [
                    "transformer.wte",
                    "transformer.wpe",
                    "transformer.drop",
                    *[f"transformer.h.{i}" for i in range(num_layers)],
                    "transformer.ln_f",
                    "score",
                ]
            ],
            True,
        ],
        [
            # Distribution correct, all layers included
            [
                ["transformer.wte", "transformer.wpe", "transformer.drop"],
                [f"transformer.h.{i}" for i in range(num_layers)],
                ["transformer.ln_f", "score"],
            ],
            True,
        ],
        [
            # Distribution correct, not all layers included
            [
                ["transformer.wte", "transformer.wpe", "transformer.drop"],
                ["transformer.h.0"],
                ["transformer.ln_f", "score"],
            ],
            False,
        ],
        [
            # First stage distribution incorrect
            [
                ["transformer.wte", "transformer.wpe"],
                [
                    "transformer.drop",
                    *[f"transformer.h.{i}" for i in range(num_layers)],
                    "transformer.ln_f",
                    "score",
                ],
            ],
            False,
        ],
        [
            # Last stage distribution incorrect
            [
                ["transformer.wte", "transformer.wpe", "transformer.drop"],
                [
                    *[f"transformer.h.{i}" for i in range(num_layers)],
                    "transformer.ln_f",
                ],
                ["score"],
            ],
            False,
        ],
    ],
)
def test_sanity_check(
    model: GPT2ForSequenceClassification,
    modules_per_stage: list[list[str]],
    expected_check_pass: bool,
):
    pipeline_template = PipelineTemplate(
        PipelineTemplate.get_model_name(model), modules_per_stage
    )

    policy = GPT2ForSequenceClassificationPolicy()
    policy.set_model(model)

    with pytest.raises(ValueError) if not expected_check_pass else nullcontext():
        policy.pipeline_template_sanity_check(pipeline_template)
