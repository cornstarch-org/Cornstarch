from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.dinov2 import Dinov2Config, Dinov2Model

from cornstarch.models.multimodal_language_model import MultimodalLanguageModelConfig


class MultimodalEncoderProjector(nn.Module):
    def __init__(self, encoder: PreTrainedModel, projection: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.config = encoder.config

    def forward(self, *args, **kwargs):
        image_feature = self.encoder(*args, **kwargs)
        image_feature = image_feature[0]
        image_feature = self.projection(image_feature)

        return image_feature


class MultimodalLanguageModel(PreTrainedModel):
    config_class = MultimodalLanguageModelConfig
    base_model_prefix = "language_model"

    def __init__(
        self,
        config: MultimodalLanguageModelConfig,
        language_model: PreTrainedModel = None,
        vision_model: MultimodalEncoderProjector = None,
    ):
        super().__init__(config)

        if language_model is None:
            language_model = AutoModelForCausalLM.from_config(config.text_config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            elif isinstance(config.vision_config, Dinov2Config):
                vision_model = Dinov2Model(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)

            if config.projection_type == "linear":
                projection = nn.Linear(
                    in_features=vision_model.config.hidden_size,
                    out_features=language_model.config.hidden_size,
                )

            vision_model = MultimodalEncoderProjector(vision_model, projection)

        self.add_module("language_model", language_model)
        self.add_module("vision_model", vision_model)

        self.language_model = language_model
        self.vision_model = vision_model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            # This is the case of first token generation
            if past_key_values is None:
                assert (
                    len(self.encoders) == 1
                ), "Currently only one encoder is supported."
                encoder = self.encoders[0]
                image_features, image_attention_mask = encoder(pixel_values)

                # Merge text and images
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids).to(input_ids.device)

                attention_mask = torch.cat(
                    [image_attention_mask, attention_mask], dim=1
                )
                position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(
                    (attention_mask == 0), 1
                )

                inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)

            # We are in the case of generation with cache
            else:
                assert (
                    input_ids.shape[1] == 1
                ), f"Expected one unprocessed token, got {input_ids.shape[1]}"

                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(
                    first_layer_past_key_value.float().sum(-2) == 0
                )

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat(
                    (extended_attention_mask, attention_mask[:, -target_length:]), dim=1
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.text_config.bos_token_id in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[
                    :, -(cache_length + input_ids.shape[1]) :
                ]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs

    @staticmethod
    def _filter_kwargs(func: Callable, **kwargs) -> dict[str, Any]:
        sig = inspect.signature(func)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    @classmethod
    def from_encoders_llm_pretrained(
        cls,
        text_model_name_or_path: str = None,
        vision_model_name_or_path: str = None,
        projection_type: str = "linear",
        **kwargs,
    ) -> MultimodalLanguageModel:
        r"""
        Example:

        ```python
        >>> # initialize a model from pretrained llama and CLIPVision models.
        >>> model = MultimodalLanguageModel.from_encoders_llm_pretrained(
        ...     encoder_names_or_paths=["openai/clip-vit-base-patch16"],
        ...     text_model_name_or_path="meta-llama/Meta-Llama-3-8B",
        ... )
        ```
        """
        language_model = AutoModelForCausalLM.from_pretrained(
            text_model_name_or_path,
            **MultimodalLanguageModel._filter_kwargs(
                AutoModel.from_pretrained, **kwargs
            ),
        )

        vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)

        if vision_config.model_type == "clip":
            vision_model = CLIPVisionModel.from_pretrained(
                vision_model_name_or_path,
                **MultimodalLanguageModel._filter_kwargs(
                    CLIPVisionModel.from_pretrained, **kwargs
                ),
            )
        elif vision_config.model_type == "dinov2":
            vision_model = Dinov2Model.from_pretrained(
                vision_model_name_or_path,
                **MultimodalLanguageModel._filter_kwargs(
                    Dinov2Model.from_pretrained, **kwargs
                ),
            )
        else:
            vision_model = AutoModel.from_pretrained(
                vision_model_name_or_path,
                **MultimodalLanguageModel._filter_kwargs(
                    AutoModel.from_pretrained, **kwargs
                ),
            )

        # TODO: need to load projection from pretrained as well
        if projection_type == "linear":
            projection = nn.Linear(
                in_features=vision_model.config.hidden_size,
                out_features=language_model.config.hidden_size,
            )

        vision_model = MultimodalEncoderProjector(vision_model, projection)

        config = MultimodalLanguageModelConfig(
            text_config=language_model.config,
            vision_config=vision_model.config,
            projection_type=projection_type,
            **kwargs,
        )

        model = cls(
            config=config, language_model=language_model, vision_model=vision_model
        )

        return model
