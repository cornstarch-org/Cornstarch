import pytest
import torch
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)

vision_token_id = 100
audio_token_id = 200

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


class TestMergeEncoderOutputsClass:
    """
    Merging encoder outputs should follow the strict data structure.

    modality_outputs: torch.Tensor (total_num_tokens, hidden_size)
    input_ids: torch.Tensor (batch_size, seq_len)
    inputs_embeds: torch.Tensor (batch_size, seq_len, hidden_size)

    where `total_num_tokens` is the sum of the number of tokens in eeach modality.
    And in input_ids, the exact same number of corresponding modality tokens should exist.
    """

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

    def build_data(
        self, vision_token_indices: list[int], audio_token_indices: list[int]
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.zeros(1, 16, dtype=torch.long)
        vision_outputs = (
            # simulate last_hidden_states in a tuple
            (torch.randn(len(vision_token_indices), 256),)
            if len(vision_token_indices) > 0
            else None
        )
        audio_outputs = (
            # simulate last_hidden_states in a tuple
            (torch.randn(len(audio_token_indices), 256),)
            if len(audio_token_indices) > 0
            else None
        )
        inputs_embeds = torch.randn(1, 16, 256)
        bitfield_attention_mask = torch.zeros(1, 16, dtype=torch.int64)

        for i in range(16):
            if i in vision_token_indices:
                input_ids[0, i] = vision_token_id
                bitfield_attention_mask[0, i] = 1 << 1
            elif i in audio_token_indices:
                input_ids[0, i] = audio_token_id
                bitfield_attention_mask[0, i] = 1 << 2
            else:
                input_ids[0, i] = i
                bitfield_attention_mask[0, i] = (
                    (1 << 62) | (1 << 0) | (1 << 1) | (1 << 2)
                )

        return {
            "input_ids": input_ids,
            "vision_outputs": vision_outputs,
            "audio_outputs": audio_outputs,
            "inputs_embeds": inputs_embeds,
            "bitfield_attention_mask": bitfield_attention_mask,
            "vision_token_indices": vision_token_indices,
            "audio_token_indices": audio_token_indices,
        }

    @pytest.mark.parametrize("data", [data1, data2, data3])
    def test_singlebatch(self, data: dict[str, list[int]]):
        """
        Check if MultimodalModel.merge_encoder_outputs generate correct results

        Encoder outputs are injected to the middle of the inputs_embeds.
        input_ids already have placeholders (tokens) for vision and audio,
        where encoder_outputs will be injected in the same location of inputs_embeds.
        """
        model: MultimodalModel = self.build_multimodal_llm()

        # Set token ids for vision and audio.
        # Do not use `set_modality_tokens()` as it requires a Processor.
        model.token_ids = {"vision": 100, "audio": 200}

        data: dict[str, torch.Tensor] = self.build_data(
            data["vision_token_indices"], data["audio_token_indices"]
        )

        input_ids: torch.Tensor = data["input_ids"]
        inputs_embeds: torch.Tensor = data["inputs_embeds"]
        bitfield_attention_mask: torch.Tensor = data["bitfield_attention_mask"]
        vision_outputs: tuple[torch.Tensor] = data["vision_outputs"]
        audio_outputs: tuple[torch.Tensor] = data["audio_outputs"]

        encoders_outputs = {}
        if vision_outputs is not None:
            encoders_outputs["vision"] = vision_outputs
        if audio_outputs is not None:
            encoders_outputs["audio"] = audio_outputs
        new_inputs_embeds, new_attention_mask = model.merge_encoder_outputs(
            encoders_outputs=encoders_outputs,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        # Compare generated inputs_embeds and attention mask with the expected one

        # inputs_embeds shape should not be changed
        assert new_inputs_embeds.shape == inputs_embeds.shape

        # Check encoder outputs are properly injected
        if vision_outputs is not None:
            assert (
                new_inputs_embeds[:, data["vision_token_indices"]][0]
                == vision_outputs[0]
            ).all()
        if audio_outputs is not None:
            assert (
                new_inputs_embeds[:, data["audio_token_indices"]][0] == audio_outputs[0]
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

        # Check newly generated attention mask
        assert (new_attention_mask == bitfield_attention_mask).all()

    @pytest.mark.parametrize(
        "data", [[data1, data2], [data2, data3], [data1, data3], [data1, data2, data3]]
    )
    def test_multibatch(self, data: list[dict[str, list[int]]]):
        model: MultimodalModel = self.build_multimodal_llm()

        # Set token ids for vision and audio.
        # Do not use `set_modality_tokens()` as it requires a Processor.
        model.token_ids = {"vision": 100, "audio": 200}

        data: list[dict[str, torch.Tensor]] = [
            self.build_data(d["vision_token_indices"], d["audio_token_indices"])
            for d in data
        ]

        vision_outputs = (
            torch.cat(
                [
                    d["vision_outputs"][0]
                    for d in data
                    if d["vision_outputs"] is not None
                ]
            ),
        )
        audio_outputs = (
            torch.cat(
                [d["audio_outputs"][0] for d in data if d["audio_outputs"] is not None]
            ),
        )

        input_ids = torch.cat([d["input_ids"] for d in data])
        inputs_embeds = torch.cat([d["inputs_embeds"] for d in data])
        bitfield_attention_mask = torch.cat(
            [d["bitfield_attention_mask"] for d in data]
        )

        new_inputs_embeds, new_attention_mask = model.merge_encoder_outputs(
            encoders_outputs={
                "vision": vision_outputs,
                "audio": audio_outputs,
            },
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        # Compare generated inputs_embeds and attention mask with the expected one

        # inputs_embeds shape should not be changed
        assert new_inputs_embeds.shape == inputs_embeds.shape

        # Check encoder outputs are properly injected
        assert (
            new_inputs_embeds[input_ids == vision_token_id] == vision_outputs[0]
        ).all()
        assert (
            new_inputs_embeds[input_ids == audio_token_id] == audio_outputs[0]
        ).all()

        # Check remaining part of the inputs_embeds are not changed
        for i in range(len(data)):
            local_input_ids = input_ids[i]
            text_indices = local_input_ids[
                (local_input_ids != vision_token_id)
                & (local_input_ids != audio_token_id)
            ]
            local_inputs_embeds = inputs_embeds[i]
            local_new_inputs_embeds = new_inputs_embeds[i]

            assert (
                local_new_inputs_embeds[text_indices]
                == local_inputs_embeds[text_indices]
            ).all()

        # Check newly generated attention mask
        assert (new_attention_mask == bitfield_attention_mask).all()
