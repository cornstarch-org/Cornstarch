from abc import ABC, abstractmethod

import pytest
import torch
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)

data1 = {
    "vision_token_indices": [3, 4, 5],
    "audio_token_indices": [8, 9, 10],
}
data2 = {
    "vision_token_indices": [6, 7, 8, 9, 10, 11, 12],
    "audio_token_indices": [],
}
data3 = {
    "vision_token_indices": [],
    "audio_token_indices": [4, 5, 6, 7, 8, 9, 10],
}


class AttentionTestClassBase(ABC):
    @abstractmethod
    def build_data(
        self, vision_token_indices: list[int], audio_token_indices: list[int]
    ) -> dict[str, torch.Tensor]: ...

    def build_multimodal_llm(self) -> MultimodalModel:
        # Copied from test_shardformer/model_zoo
        vision_config = CLIPVisionConfig(
            hidden_size=256,
            intermediate_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            use_cache=False,
        )
        audio_config = WhisperConfig(
            max_source_positions=64,
            max_target_positions=64,
            d_model=64,
            encoder_ffn_dim=64,
            encoder_attention_heads=16,
            encoder_layers=4,
            is_encoder_decoder=False,
            use_cache=False,
        )
        language_config = LlamaConfig(
            hidden_size=512,
            intermediate_size=64,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_hidden_layers=4,
            use_cache=False,
        )

        return MultimodalModel(
            encoders={
                "vision": ModalEncoderModule(CLIPVisionModel(vision_config)),
                "audio": ModalEncoderModule(WhisperEncoder(audio_config)),
            },
            language_model=LlamaForCausalLM(language_config),
        )

    @abstractmethod
    def test(self): ...


class TestEncoderInjectionAttentionClass(AttentionTestClassBase):
    def build_data(
        self, vision_token_indices: list[int], audio_token_indices: list[int]
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.zeros(1, 16, dtype=torch.long)
        vision_outputs = (torch.randn(1, len(vision_token_indices), 256),)
        audio_outputs = (torch.randn(1, len(audio_token_indices), 256),)
        inputs_embeds = torch.randn(1, 16, 256)
        bit_attention_mask = torch.zeros(1, 16, dtype=torch.int64)
        full_attention_mask = torch.zeros(1, 16, 16, dtype=torch.bool)

        for i in range(16):
            if i in vision_token_indices:
                input_ids[0, i] = 100
                bit_attention_mask[0, i] = 1 << 0
                full_attention_mask[0, i, vision_token_indices] = 1
            elif i in audio_token_indices:
                input_ids[0, i] = 200
                bit_attention_mask[0, i] = 1 << 1
                full_attention_mask[0, i, audio_token_indices] = 1
            else:
                input_ids[0, i] = i
                bit_attention_mask[0, i] = (1 << 62) | (1 << 0) | (1 << 1)
                full_attention_mask[0, i, : i + 1] = 1

        return {
            "input_ids": input_ids,
            "vision_outputs": vision_outputs,
            "audio_outputs": audio_outputs,
            "inputs_embeds": inputs_embeds,
            "causal_attention_mask": torch.ones(1, 16, dtype=torch.bool),
            "bit_attention_mask": bit_attention_mask,
            "full_attention_mask": full_attention_mask,
            "vision_token_indices": vision_token_indices,
            "audio_token_indices": audio_token_indices,
        }

    @pytest.mark.parametrize("data", [data1, data2, data3])
    @pytest.mark.parametrize("attention_type", ["causal", "bitattn", "full"])
    def test(self, data: dict[str, list[int]], attention_type: str):
        """
        Check if MultimodalModel.merge_encoder_outputs generate correct results

        Encoder outputs are injected to the middle of the inputs_embeds.
        The length of sequence length and labels does not change.

        For attention mask generation, it depends on the arguments.
        1. If 2D attention mask (bool or 1/0 type, shape: [batch_size, seq_len]) is passed,
            it should be returned as is.
            It will be updated to 4D causal mask in the model forward method.
            This is for backward compatibility.
        2. If 3D attention mask (bool or 1/0 type, shape: [batch_size, seq_len, seq_len]) is passed,
            it should be returned as is.
            It will be converted to block mask and used in FlexAttention.
        3. If None is passed,
            2D bit attention mask (int64 type, shape: [batch_size, sqe_len]) will be generated.
            It will be converted to block mask and used in FlexAttention.
        """
        model: MultimodalModel = self.build_multimodal_llm()
        model.set_token_ids({"vision": 100, "audio": 200})

        data: dict[str, torch.Tensor] = self.build_data(
            data["vision_token_indices"], data["audio_token_indices"]
        )

        attention_mask = None
        if attention_type == "causal":
            attention_mask = data["causal_attention_mask"]
        elif attention_type == "full":
            attention_mask = data["full_attention_mask"]

        labels: torch.Tensor = data["input_ids"].clone()
        input_ids: torch.Tensor = data["input_ids"]
        inputs_embeds: torch.Tensor = data["inputs_embeds"]
        vision_outputs: tuple[torch.Tensor] = data["vision_outputs"]
        audio_outputs: tuple[torch.Tensor] = data["audio_outputs"]

        new_inputs_embeds, new_attention_mask, new_position_ids, new_labels = (
            model.merge_encoder_outputs(
                encoder_inputs={
                    "vision": None,
                    "audio": None,
                },
                encoder_outputs={
                    "vision": vision_outputs,
                    "audio": audio_outputs,
                },
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        )

        # Expect new inputs_embeds does not increase the length of the sequence
        assert inputs_embeds.shape == new_inputs_embeds.shape
        assert labels.shape == new_labels.shape

        # Check encoder outputs properly injected
        assert (
            new_inputs_embeds[:, data["vision_token_indices"]] == vision_outputs[0]
        ).all()
        assert (
            new_inputs_embeds[:, data["audio_token_indices"]] == audio_outputs[0]
        ).all()

        # Check remaining part of the inputs_embeds are not changed
        text_indices = [
            i
            for i in range(16)
            if i not in data["vision_token_indices"]
            and i not in data["audio_token_indices"]
        ]
        assert (
            new_inputs_embeds[:, text_indices] == inputs_embeds[:, text_indices]
        ).all()

        # check attention mask
        if attention_type == "causal" or attention_type == "full":
            assert (new_attention_mask == attention_mask).all()
        else:
            assert (new_attention_mask == data["bit_attention_mask"]).all()


class TestEncoderPrependedAttentionClass(AttentionTestClassBase):
    def build_data(
        self, vision_token_indices: list[int], audio_token_indices: list[int]
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.arange(16, dtype=torch.long).unsqueeze(0).repeat(1, 1)
        vision_outputs = (torch.randn(1, len(vision_token_indices), 256),)
        audio_outputs = (torch.randn(1, len(audio_token_indices), 256),)
        inputs_embeds = torch.randn(1, 16, 256)
        prepended_seq_length = 16 + len(vision_token_indices) + len(audio_token_indices)
        bit_attention_mask = torch.zeros(1, prepended_seq_length, dtype=torch.int64)
        full_attention_mask = torch.zeros(
            1, prepended_seq_length, prepended_seq_length, dtype=torch.bool
        )

        for i in range(16):
            offset = len(vision_token_indices) + len(audio_token_indices)
            bit_attention_mask[0, offset + i] = (1 << 62) | (1 << 0) | (1 << 1)
            full_attention_mask[0, offset + i, : offset + i] = 1

        for i in range(len(vision_token_indices)):
            offset = 0
            bit_attention_mask[0, offset + i] = 1 << 0

        for i in range(len(audio_token_indices)):
            offset = len(vision_token_indices)
            bit_attention_mask[0, offset + i] = 1 << 1

        return {
            "input_ids": input_ids,
            "vision_outputs": vision_outputs,
            "audio_outputs": audio_outputs,
            "inputs_embeds": inputs_embeds,
            "causal_attention_mask": torch.ones(1, 16, dtype=torch.bool),
            "bit_attention_mask": bit_attention_mask,
            "full_attention_mask": full_attention_mask,
            "vision_token_indices": vision_token_indices,
            "audio_token_indices": audio_token_indices,
        }

    @pytest.mark.parametrize("data", [data1, data2, data3])
    @pytest.mark.parametrize("attention_type", ["causal", "bitattn", "full"])
    def test(self, data: dict[str, list[int]], attention_type: str):
        """
        Check if MultimodalModel.merge_encoder_outputs generate correct results

        Encoder outputs are prepended to the original inputs_embeds.
        The length of sequence length and labels are increased by the number of tokens in the encoder outputs.

        For attention mask generation, it depends on the arguments.
        1. If 2D attention mask (bool or 1/0 type, shape: [batch_size, seq_len]) is passed,
            it should be returned as is.
            It will be updated to 4D causal mask in the model forward method.
            This is for backward compatibility.
        2. If 3D attention mask (bool or 1/0 type, shape: [batch_size, seq_len, seq_len]) is passed,
            it should be returned as is.
            It will be converted to block mask and used in FlexAttention.
        3. If None is passed,
            2D bit attention mask (int64 type, shape: [batch_size, sqe_len]) will be generated.
            It will be converted to block mask and used in FlexAttention.
        """
        model: MultimodalModel = self.build_multimodal_llm()

        vision_token_indices = data["vision_token_indices"]
        audio_token_indices = data["audio_token_indices"]

        data: dict[str, torch.Tensor] = self.build_data(
            vision_token_indices, audio_token_indices
        )

        attention_mask = None
        if attention_type == "causal":
            attention_mask = data["causal_attention_mask"]
        elif attention_type == "full":
            attention_mask = data["full_attention_mask"]

        labels: torch.Tensor = data["input_ids"].clone()
        input_ids: torch.Tensor = data["input_ids"]
        inputs_embeds: torch.Tensor = data["inputs_embeds"]
        vision_outputs: tuple[torch.Tensor] = data["vision_outputs"]
        audio_outputs: tuple[torch.Tensor] = data["audio_outputs"]

        new_inputs_embeds, new_attention_mask, new_position_ids, new_labels = (
            model.merge_encoder_outputs(
                encoder_inputs={
                    "vision": None,
                    "audio": None,
                },
                encoder_outputs={
                    "vision": vision_outputs,
                    "audio": audio_outputs,
                },
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        )

        # For prepended, "causal" or "full" attention mask already raise ValueError.
        # Here, we only check the case of bit attention mask.
        prepended_seq_length = 16 + len(vision_token_indices) + len(audio_token_indices)
        assert new_inputs_embeds.shape[1] == prepended_seq_length
        assert new_labels.shape[1] == prepended_seq_length

        # token order: vision - audio - text
        assert (
            new_inputs_embeds[:, : len(vision_token_indices)] == vision_outputs[0]
        ).all()

        assert (
            new_inputs_embeds[
                :,
                len(vision_token_indices) : len(vision_token_indices)
                + len(audio_token_indices),
            ]
            == audio_outputs[0]
        ).all()

        assert (
            new_inputs_embeds[:, len(vision_token_indices) + len(audio_token_indices) :]
            == inputs_embeds
        ).all()

        # check attention mask
        if attention_type == "causal":
            assert (new_attention_mask == torch.ones(1, prepended_seq_length)).all()
        elif attention_type == "full":
            assert (new_attention_mask == attention_mask).all()
        else:
            assert (new_attention_mask == data["bit_attention_mask"]).all()
