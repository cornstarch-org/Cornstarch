"""Data-side context-parallel metadata at the multimodal merge boundary."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from cornstarch.models.multimodal.execution import merge_modality_encoder_outputs


class _LanguageStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre_decoder = nn.ModuleDict({"embed_tokens": nn.Embedding(128, 2)})
        self.pre_decoder["embed_tokens"].weight.data.fill_(1.0)


def test_cp_merge_selects_features_for_local_placeholder_positions() -> None:
    """A CP rank receives only the full encoder features for placeholders it owns."""
    model = _LanguageStub()
    global_ids = torch.tensor([[99, 1, 99, 2]])
    position_ids = torch.tensor([[1, 2]])
    local_ids = global_ids.index_select(1, position_ids[0])
    labels = local_ids.clone()
    features = BaseModelOutput(
        last_hidden_state=torch.tensor([[[10.0, 11.0], [20.0, 21.0]]])
    )

    merged = merge_modality_encoder_outputs(
        language_model=model,
        input_ids=local_ids,
        labels=labels,
        encoder_outputs={"vision": features},
        modality_token_ids={"vision": 99},
        language_model_inputs={
            "position_ids": position_ids,
            "cp_global_input_ids": global_ids,
            "shift_labels": torch.tensor([[99, 2]]),
            "num_items_in_batch": torch.tensor(3),
        },
    )

    torch.testing.assert_close(merged["inputs_embeds"][0, 0], torch.ones(2))
    torch.testing.assert_close(
        merged["inputs_embeds"][0, 1], torch.tensor([20.0, 21.0])
    )
    assert torch.equal(merged["position_ids"], position_ids)
    assert int(merged["num_items_in_batch"]) == 3
    assert "cp_global_input_ids" not in merged
