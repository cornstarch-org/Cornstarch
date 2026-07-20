from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from examples.common import configure_special_tokens
from examples.distributed.common import (
    local_trainable_parameters,
    microbatch_collate,
    move_microbatches_to_device,
)


class _Tokenizer:
    def __init__(self, *, eos_token: str | None = "</s>") -> None:
        self.eos_token = eos_token
        self.pad_token: str | None = None
        self._tokens = {"a": 0, "b": 1, "</s>": 2}

    def __len__(self) -> int:
        return len(self._tokens)

    def add_special_tokens(self, values: dict[str, list[str]]) -> None:
        for token in values["additional_special_tokens"]:
            self._tokens.setdefault(token, len(self._tokens))

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens.get(token, -1)


def test_configure_special_tokens_resizes_config_before_model_build() -> None:
    config = PretrainedConfig(vocab_size=3)
    tokenizer = _Tokenizer()

    token_ids = configure_special_tokens(config, tokenizer, ["<image>", "<audio>"])

    assert tokenizer.pad_token == tokenizer.eos_token
    assert config.vocab_size == len(tokenizer) == 5
    assert token_ids == {"<image>": 3, "<audio>": 4}


def test_configure_special_tokens_requires_padding_source() -> None:
    with pytest.raises(ValueError, match="EOS or padding token"):
        configure_special_tokens(PretrainedConfig(), _Tokenizer(eos_token=None), ["<image>"])


def test_microbatch_collate_returns_exact_equal_splits() -> None:
    samples = [{"value": torch.tensor(index)} for index in range(6)]

    microbatches = microbatch_collate(3)(samples)

    assert len(microbatches) == 3
    assert [batch["value"].tolist() for batch in microbatches] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]


def test_microbatch_collate_rejects_non_divisible_batch() -> None:
    samples = [{"value": torch.tensor(index)} for index in range(5)]

    with pytest.raises(ValueError, match="must be nonzero and divisible"):
        microbatch_collate(2)(samples)


def test_move_microbatches_to_device_preserves_non_tensors() -> None:
    microbatches = [{"value": torch.tensor([1]), "metadata": "sample"}]

    moved = move_microbatches_to_device(microbatches, torch.device("cpu"))

    assert moved[0]["value"].device.type == "cpu"
    assert moved[0]["metadata"] == "sample"


def test_local_trainable_parameters_excludes_frozen_and_meta_parameters() -> None:
    local = nn.Linear(2, 2)
    local.bias.requires_grad_(False)
    remote = nn.Linear(2, 2, device="meta")

    parameters = local_trainable_parameters(local, remote)

    assert parameters == [local.weight]
