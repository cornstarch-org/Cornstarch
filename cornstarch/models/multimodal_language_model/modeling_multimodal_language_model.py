from __future__ import annotations

import functools
import inspect
import warnings
from types import MethodType
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from transformers.activations import get_activation
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.llava.modeling_llava import (
    LlavaConfig,
    LlavaForConditionalGeneration,
)
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    image_size_to_num_patches,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
)

from cornstarch.models.internvl2.modeling_internvl_chat import (
    InternVLChatConfig,
    InternVLChatModel,
)
from cornstarch.models.multimodal_language_model import MultimodalProjectorConfig


def prepend_modal_output_to_inputs_embeds(
    inputs: dict,
    output: BaseModelOutput | tuple,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    pad_token_id: int = -1,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple postprocess_projector_callback that prepends the output of the `ModalEncoderModule` to the `inputs_embeds`."""
    # output[0] == output.last_hidden_state
    new_inputs_embeds = torch.cat([output[0], inputs_embeds], dim=1)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids).to(input_ids.device)

    new_attention_mask = torch.cat(
        [
            torch.ones(output[0].shape[:2], device=attention_mask.device),
            attention_mask,
        ],
        dim=1,
    ).to(dtype=torch.long)
    new_position_ids = (
        (new_attention_mask.cumsum(-1) - 1)
        .masked_fill_((new_attention_mask == 0), 1)
        .to(dtype=torch.long)
    )

    if labels is not None:
        new_labels = torch.full(
            new_inputs_embeds.shape[:2], ignore_index, device=labels.device
        )
        new_labels[:, -labels.shape[1] :] = labels
    else:
        new_labels = None

    return (
        new_inputs_embeds,
        new_attention_mask,
        new_position_ids,
        new_labels,
    )


class PretrainedVisionLanguageModel:
    def from_pretrained(self, *args, **kwargs) -> MultimodalModel:
        pass

    def preprocess_vision_callback(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return inputs

    def postprocess_vision_callback(
        self, model: PreTrainedModel, inputs: dict, output: BaseModelOutput | tuple
    ) -> BaseModelOutput | tuple:
        return output

    def postprocess_projector_callback(
        self,
        model: PreTrainedModel,
        inputs: dict,
        output: BaseModelOutput | tuple,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int = -1,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return prepend_modal_output_to_inputs_embeds(
            inputs,
            output,
            input_ids,
            inputs_embeds,
            attention_mask,
            labels,
            pad_token_id,
            ignore_index,
        )


class LlavaModel(PretrainedVisionLanguageModel):
    """A class for Llava pretrained models.
    This is only for Llava <= 1.5, not compatible with Llava 1.6 (Llava-Next)"""

    def __init__(self, config: LlavaConfig):
        self.config = config

    def from_pretrained(self, *args, **kwargs) -> MultimodalModel:
        model: LlavaForConditionalGeneration = (
            LlavaForConditionalGeneration.from_pretrained(
                self.config.name_or_path, config=self.config, *args, **kwargs
            )
        )
        model.vision_tower.config.output_hidden_states = True
        vision_encoder = model.vision_tower
        language_model = model.language_model
        language_model.config.pad_token_id = (
            model.config.pad_token_id if model.config.pad_token_id is not None else -1
        )

        # Create projector
        projector = model.multi_modal_projector
        projector_config = MultimodalProjectorConfig(
            encoder_config=vision_encoder.config,
            text_config=language_model.config,
            projection_type="mlp",
        )
        vision_projector = MultimodalProjector(projector_config, projector)

        vision_tower = ModalEncoderModule(
            model=vision_encoder,
            projector=vision_projector,
            postprocess_module_callback=functools.partial(
                self.postprocess_vision_callback, model=model
            ),
            postprocess_projector_callback=functools.partial(
                self.postprocess_projector_callback, model=model
            ),
        )

        return MultimodalModel(
            encoders={"vision": vision_tower},
            language_model=language_model,
        )

    @staticmethod
    def postprocess_vision_callback(
        model: LlavaForConditionalGeneration,
        inputs: dict,
        output: BaseModelOutput | tuple,
    ) -> BaseModelOutput | tuple:
        config: LlavaConfig = model.config
        vision_feature_layer = config.vision_feature_layer
        vision_feature_select_strategy = config.vision_feature_select_strategy

        if isinstance(output, ModelOutput):
            if output.hidden_states is None:
                # vision_tower is executed without output_hidden_states=True.
                # Use the last_hidden_state.
                selected_image_feature = output.last_hidden_state
            else:
                selected_image_feature = output.hidden_states[vision_feature_layer]
        else:
            if len(output) == 1 or output[1] is None:
                selected_image_feature = output[0]
            else:
                selected_image_feature = output[1][vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {vision_feature_select_strategy}"
            )

        output.last_hidden_state = selected_image_feature
        return output

    @staticmethod
    def postprocess_projector_callback(
        model: LlavaForConditionalGeneration,
        inputs: dict,
        output: BaseModelOutput | tuple,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        config: LlavaConfig = model.config
        if "pixel_values" not in inputs or inputs["pixel_values"] is None:
            return inputs_embeds, attention_mask, None, labels

        pixel_values = inputs["pixel_values"]
        image_features = output[0]
        past_key_values = (
            inputs["past_key_values"] if "past_key_values" in inputs else None
        )

        legacy_processing = (
            (input_ids == config.image_token_index).sum(1).max()
            < config.image_seq_length
        ) or (input_ids.shape[-1] == 1 and pixel_values is not None)

        if legacy_processing:
            from transformers.models.llava.modeling_llava import logger

            logger.warning_once(
                "Expanding inputs for image tokens in LLaVa should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            )
            # prefill stage vs decoding stage (legacy behavior copied)
            if input_ids.shape[1] != 1:
                inputs_embeds, attention_mask, labels, position_ids = (
                    model._merge_input_ids_with_image_features(
                        image_features, inputs_embeds, input_ids, attention_mask, labels
                    )
                )
                cache_position = torch.arange(
                    attention_mask.shape[1], device=attention_mask.device
                )
            else:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
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
                cache_position = torch.arange(
                    attention_mask.shape[1], device=attention_mask.device
                )[-target_length:]

        # TODO: @raushan retain only the new behavior after v4.47
        else:
            special_image_mask = (
                (input_ids == config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        return inputs_embeds, attention_mask, None, labels


class LlavaNextModel:
    """A class for Llava-Next (1.6) pretrained models.

    Code borrowed from https://github.com/huggingface/transformers/blob/v4.42.3/src/transformers/models/llava_next/modeling_llava_next.py
    """

    def __init__(self, config: LlavaNextConfig):
        self.config = config

    def from_pretrained(self, *args, **kwargs) -> MultimodalModel:
        model: LlavaNextForConditionalGeneration = (
            LlavaNextForConditionalGeneration.from_pretrained(
                self.config.name_or_path, config=self.config, *args, **kwargs
            )
        )

        model.vision_tower.config.output_hidden_states = True
        vision_encoder = model.vision_tower
        language_model = model.language_model
        language_model.config.pad_token_id = (
            model.config.pad_token_id if model.config.pad_token_id is not None else -1
        )

        # Create projector
        projector = model.multi_modal_projector
        projector_config = MultimodalProjectorConfig(
            encoder_config=vision_encoder.config,
            text_config=language_model.config,
            projection_type="mlp",
        )
        vision_projector = MultimodalProjector(projector_config, projector)

        vision_tower = ModalEncoderModule(
            model=vision_encoder,
            projector=vision_projector,
            additional_args=["image_sizes"],
            preprocess_callback=functools.partial(
                self.preprocess_vision_callback, model=model
            ),
            postprocess_module_callback=functools.partial(
                self.postprocess_vision_callback, model=model
            ),
            postprocess_projector_callback=functools.partial(
                self.postprocess_projector_callback, model=model
            ),
        )

        mm_model = MultimodalModel(
            encoders={"vision": vision_tower},
            language_model=language_model,
        )

        mm_model.image_newline = model.image_newline
        return mm_model

    @staticmethod
    def preprocess_vision_callback(
        model: LlavaNextForConditionalGeneration, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["image_sizes"]

        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=model.config.image_grid_pinpoints,
                patch_size=model.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        # figure out if pixel_values is concatenated or stacked
        if pixel_values.dim() == 5:
            # stacking when input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [
                pix_val[:num_patch]
                for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(
                f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions"
            )

        inputs["pixel_values"] = pixel_values
        return inputs

    @staticmethod
    def postprocess_vision_callback(
        model: LlavaNextForConditionalGeneration,
        inputs: dict,
        output: BaseModelOutput | tuple,
    ) -> BaseModelOutput | tuple:
        vision_feature_layer = model.config.vision_feature_layer
        vision_feature_select_strategy = model.config.vision_feature_select_strategy

        if isinstance(output, ModelOutput):
            if output.hidden_states is None:
                # vision_tower is executed without output_hidden_states=True.
                # Use the last_hidden_state.
                selected_image_feature = output.last_hidden_state
            else:
                selected_image_feature = output.hidden_states[vision_feature_layer]
        else:
            if len(output) == 1 or output[1] is None:
                selected_image_feature = output[0]
            else:
                selected_image_feature = output[1][vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {vision_feature_select_strategy}"
            )

        output.last_hidden_state = selected_image_feature
        return output

    @staticmethod
    def postprocess_projector_callback(
        model: LlavaNextForConditionalGeneration,
        inputs: dict,
        output: BaseModelOutput | tuple,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        config: LlavaNextConfig = model.config
        image_sizes = inputs["image_sizes"]

        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=config.image_grid_pinpoints,
                patch_size=config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        # output[0] == output.last_hidden_state
        image_features = output[0]
        image_features = torch.split(image_features, image_num_patches, dim=0)

        # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
        image_features, feature_lens = model.pack_image_features(
            image_features,
            image_sizes,
            image_newline=model.image_newline,
            vision_feature_select_strategy=config.vision_feature_select_strategy,
        )

        inputs_embeds = inputs_embeds.to(image_features.dtype)
        inputs_embeds, attention_mask, position_ids, labels, _ = (
            model._merge_input_ids_with_image_features(
                image_features,
                feature_lens,
                inputs_embeds,
                input_ids,
                attention_mask,
                position_ids=None,
                labels=labels,
                image_token_index=config.image_token_index,
                ignore_index=config.ignore_index,
            )
        )

        return inputs_embeds, attention_mask, position_ids, labels


class InternVL2Model(PretrainedVisionLanguageModel):
    """A class for InternVL2 pretrained models."""

    def __init__(self, config: InternVLChatConfig):
        self.config = config

    def from_pretrained(self, *args, **kwargs) -> MultimodalModel:
        model: InternVLChatModel = InternVLChatModel.from_pretrained(
            self.config.name_or_path, config=self.config, *args, **kwargs
        )
        vision_encoder = model.vision_model
        language_model = model.language_model

        # Create a projector
        projector = model.mlp1
        projector_config = MultimodalProjectorConfig(
            encoder_config=vision_encoder.config,
            text_config=language_model.config,
            projection_type="mlp",
        )
        vision_projector = MultimodalProjector(projector_config, projector)

        vision_encoder = ModalEncoderModule(
            model=vision_encoder,
            projector=vision_projector,
            postprocess_module_callback=functools.partial(
                self.postprocess_vision_callback, model=model
            ),
            postprocess_projector_callback=functools.partial(
                self.postprocess_projector_callback, model=model
            ),
        )

        mm_model = MultimodalModel(
            encoders={"vision": vision_encoder},
            language_model=language_model,
        )

        mm_model.chat = MethodType(InternVLChatModel.chat, model)
        mm_model.batch_chat = MethodType(InternVLChatModel.batch_chat, model)
        model.generate = MethodType(
            functools.partial(MultimodalModel.generate, add_input_ids=False), mm_model
        )

        return mm_model

    @staticmethod
    def postprocess_vision_callback(
        model: InternVLChatModel,
        inputs: dict,
        output: BaseModelOutput | tuple,
    ) -> BaseModelOutput | tuple:
        if model.select_layer == -1:
            vit_embeds = output.last_hidden_state
        else:
            vit_embeds = output.hidden_states[model.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = model.pixel_shuffle(
            vit_embeds, scale_factor=model.downsample_ratio
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        output.last_hidden_state = vit_embeds
        return output

    @staticmethod
    def postprocess_projector_callback(
        model: InternVLChatModel,
        inputs: dict,
        output: BaseModelOutput | tuple,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vit_embeds = output[0]

        B, N, C = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == model.img_context_token_id
        try:
            inputs_embeds[selected] = inputs_embeds[
                selected
            ] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception:
            vit_embeds = vit_embeds.reshape(-1, C)
            n_token = selected.sum()
            inputs_embeds[selected] = (
                inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]
            )

        inputs_embeds = inputs_embeds.reshape(B, N, C)

        return inputs_embeds, attention_mask, None, labels


class Qwen2VLModel(PretrainedVisionLanguageModel):
    """A class for QWen2VL pretrained models."""

    @staticmethod
    def vision_transformer_forward(
        self: Qwen2VisionTransformerPretrainedModel,
        original_forward: Callable,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        """
        Wrapper function for the forward method of Qwen2VL vision transformer.
        This is for backward compatibility of a few additional HF arguments.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states = original_forward(hidden_states, grid_thw)

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutput(last_hidden_state=hidden_states)

    class FakeMerger(nn.Module):
        """Merger is merged into Qwen2VisionTransformer.

        As Cornstarch manages them separately, we need to fake the merger layer
        in the vision transformer.
        This does nothing in forward.
        """

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    def __init__(self, config: Qwen2VLConfig):
        self.config = config

    def from_pretrained(self, *args, **kwargs) -> MultimodalModel:
        model: Qwen2VLForConditionalGeneration = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.name_or_path, config=self.config, *args, **kwargs
            )
        )

        vision_encoder = model.visual
        vision_encoder.main_input_name = "pixel_values"

        # Qwen2VL vision encoder has an embedded MLP layer. Split it.
        projector = vision_encoder.merger
        vision_encoder.merger = self.FakeMerger()
        vision_encoder.forward = MethodType(
            functools.partial(
                Qwen2VLModel.vision_transformer_forward,
                original_forward=vision_encoder.forward,
            ),
            vision_encoder,
        )
        projector_config = MultimodalProjectorConfig(
            encoder_config=vision_encoder.config,
            text_config=model.config,
            projection_type="mlp",
        )
        vision_projector = MultimodalProjector(projector_config, projector)

        vision_encoder = ModalEncoderModule(
            model=vision_encoder,
            projector=vision_projector,
            additional_args=[
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            ],
            preprocess_callback=functools.partial(
                self.preprocess_vision_callback, visual_dtype=model.visual.get_dtype()
            ),
            postprocess_projector_callback=functools.partial(
                self.postprocess_projector_callback, model=model
            ),
        )

        delattr(model, "visual")
        mm_model = MultimodalModel(
            encoders={"vision": vision_encoder},
            language_model=model,
        )

        return mm_model

    def preprocess_vision_callback(
        self, visual_dtype: torch.dtype, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        new_inputs = {}

        if "pixel_values" in inputs:
            new_inputs["hidden_states"] = inputs["pixel_values"]
            new_inputs["grid_thw"] = inputs["image_grid_thw"]
        if "pixel_values_videos" in inputs:
            new_inputs["hidden_states"] = inputs["pixel_values_videos"]
            new_inputs["grid_thw"] = inputs["video_grid_thw"]

        new_inputs["hidden_states"] = new_inputs["hidden_states"].to(visual_dtype)
        return new_inputs

    def postprocess_projector_callback(
        self,
        model: PreTrainedModel,
        inputs: dict,
        output: BaseModelOutput | tuple,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if "pixel_values" in inputs:
            image_mask = input_ids == model.config.image_token_id
            inputs_embeds[image_mask] = output[0]
        if "pixel_values_videos" in inputs:
            video_mask = input_ids == model.config.video_token_id
            inputs_embeds[video_mask] = output[0]

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        return inputs_embeds, attention_mask, None, labels


class MultimodalProjector(PreTrainedModel):
    """
    An abstract class to handle weights initialization of projector layers
    between encoders and a language model.
    """

    config_class = MultimodalProjectorConfig
    base_model_prefix = ""
    main_input_name = "inputs_embeds"
    supports_gradient_checkpointing = True

    config: MultimodalProjectorConfig

    def __init__(
        self,
        config: MultimodalProjectorConfig,
        projection: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        self.gradient_checkpointing = False

        if projection:
            self.projection = projection
        else:
            if config.projection_type == "linear":
                self.projection = nn.Linear(
                    in_features=config.in_features,
                    out_features=config.out_features,
                )
            elif config.projection_type == "mlp":
                self.projection = nn.Sequential(
                    nn.Linear(
                        in_features=config.in_features,
                        out_features=config.out_features,
                    ),
                    get_activation(config.activation),
                    nn.Linear(
                        in_features=config.out_features,
                        out_features=config.out_features,
                    ),
                )
            elif config.projection_type == "qformer":
                raise NotImplementedError

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, inputs_embeds: torch.Tensor, return_dict: bool = True
    ) -> Union[ModelOutput, tuple]:
        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(self.projection, inputs_embeds)
        else:
            outputs = self.projection(inputs_embeds)

        if not return_dict:
            return tuple(outputs)

        return ModelOutput(hidden_states=outputs)


class ModalModuleBase(nn.Module):
    def __init__(
        self, model: PreTrainedModel, projector: Optional[MultimodalProjector] = None
    ):
        super().__init__()
        self.module = model
        self.projector = projector
        self.config = (model.config, projector.config if projector else None)

        if projector is not None:
            if isinstance(self, ModalEncoderModule):
                assert projector.config.in_features == model.config.hidden_size, (
                    f"Input features of projector ({projector.config.in_features}) "
                    f"should be equal to hidden size of model ({model.config.hidden_size})."
                )
            elif isinstance(self, ModalDecoderModule):
                assert projector.config.out_features == model.config.hidden_size, (
                    f"Output features of projector ({projector.config.out_features}) "
                    f"should be equal to hidden size of model ({model.config.hidden_size})."
                )
            else:
                raise ValueError(
                    "ModalModule should be either ModalEncoderModule or ModalDecoderModule."
                )

    def train(self, module: False = True, projector: bool = True) -> ModalModuleBase:
        self.module.train(module)
        if self.projector:
            self.projector.train(projector)
        return self

    def get_modules(self) -> list[nn.Module]:
        modules = [self.module]
        if self.projector is not None:
            modules.append(self.projector)
        return modules


class ModalEncoderModule(ModalModuleBase):

    def __init__(
        self,
        model: PreTrainedModel,
        projector: Optional[MultimodalProjector] = None,
        additional_args: list[str] = [],
        preprocess_callback: Callable[
            [dict[str, Any]], dict[str, Any]
        ] = lambda inputs: inputs,
        postprocess_module_callback: Callable[
            [dict, BaseModelOutput | tuple], BaseModelOutput | tuple
        ] = lambda inputs, output: output,
        postprocess_projector_callback: Callable[
            [
                dict,
                BaseModelOutput | tuple,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            torch.Tensor,
        ] = prepend_modal_output_to_inputs_embeds,
    ):
        """
        A wrapper module for encoder model with a projector layer.

        Args:
            model (`PreTrainedModel`):
                An encoder model.
            projector (`MultimodalProjector`, *optional*):
                A projector layer.
                If not given, this `ModalEncoderModule` cannot be attached to `MutlimodalModel`.
            additional_args (`list[str]`, *optional*): A list of required arguments for the encoder module.
                `MultimodalModel` will automatically infer required arguments to be passed by calling
                `signature.inspect()` on model's forward method.
                However, if additional arguments not used in the forward method but necessary in processing,
                they can be passed here.
                Arguments, if given, are used by preprocess_callback and filtered out.
            preprocess_callback (`Callable[[dict[str, Any]], dict[str, Any]]`, *optional*):
                A function to preprocess inputs.
                Called before the encoder module is called to manipulate the inputs. Default is an identity function.
            postprocess_module_callback (`Callable[[dict, BaseModelOutput | tuple], BaseModelOutput]`, *optional*):
                A function to postprocess the output of the encoder module.
                Called after the encoder module is called and before the projector is called. Default is an identity function.
                Inputs to the callback:
                    dict: Inputs to the encoder module.
                    BaseModelOutput | tuple: Output of the encoder module.
            postprocess_projector_callback (`Callable[[dict, BaseModelOutput | tuple, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]`, *optional*):
                A function to postprocess the output of the `ModalEncoderModule`.
                Called after the encoder module and the projector are called.
                Second argument of the function is the original `inputs_embeds` from the language model backbone.
                After manipulating the `inputs_embeds`, returned `torch.Tensor` will replace the original `inputs_embeds`.
                This function is called by `MultimodalModel` after the encoder is called.
                When there are multiple encoders, the order of the function call is the reverse order of the encoder modules.
                Inputs to the callback:
                    dict: Inputs to the encoder module.
                    BaseModelOutput | tuple: Output of the encoder module.
                    torch.Tensor: `input_ids` from the language model backbone.
                    torch.Tensor: `inputs_embeds` from the language model backbone.
                    torch.Tensor: `attention_mask` from the language model backbone.
                    torch.Tensor: `labels` from the language model backbone.
                Outputs from the callback:
                    torch.Tensor: new `inputs_embeds` to be used in the language model backbone.
                    torch.Tensor: new `attention_mask` to be used in the language model backbone.
                    torch.Tensor: new `position_ids` to be used in the language model backbone.
                    torch.Tensor: new `labels` to be used in the language model backbone.
        """
        super().__init__(model, projector)
        self.additional_args = additional_args
        self.preprocess_callback = preprocess_callback
        self.postprocess_module_callback = postprocess_module_callback
        self.postprocess_projector_callback = postprocess_projector_callback

    def forward(
        self,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> ModelOutput | tuple:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.module.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.module.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.module.config.output_hidden_states
        )

        # Merge args to kwargs
        module_params = list(inspect.signature(self.module.forward).parameters.keys())
        args_dict = {param_name: arg for param_name, arg in zip(module_params, args)}
        kwargs.update(args_dict)

        # Call preprocess callback
        kwargs = self.preprocess_callback(inputs=kwargs)

        # Filter out additional arguments
        kwargs = {k: v for k, v in kwargs.items() if k in module_params}

        outputs: BaseModelOutput = self.module(
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        outputs = self.postprocess_module_callback(inputs=kwargs, output=outputs)

        if self.projector is None:
            return outputs

        # `postprocess_projector_callback`` cannot be called here, since we do not have language model data.
        # It will be called from `MultimodalModel`.
        return self.projector(outputs[0], return_dict=return_dict)


class ModalDecoderModule(ModalModuleBase):
    # TODO: support callbacks like `ModalEncoderModule`
    def __init__(self, model: PreTrainedModel, projector: MultimodalProjector):
        super().__init__(model, projector)

    def forward(
        self, return_dict: Optional[bool] = None, *args, **kwargs
    ) -> ModelOutput | tuple:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.module.config.use_return_dict
        )

        if self.projector is None:
            return self.module(return_dict=return_dict, *args, **kwargs)

        return self.module(
            self.projector(return_dict=return_dict, *args, **kwargs)[0],
            return_dict=return_dict,
        )


class MultimodalModel(nn.Module):
    def __init__(
        self,
        encoders: dict[str, ModalEncoderModule],
        language_model: Optional[PreTrainedModel] = None,
        init_projector_type: str = "linear",
        init_activation: str = "gelu",
    ):
        """
        A representation of multimodal model, with arbitrary number of
        different types of encoders, and an optional large language model.

        Args:
            encoders (`dict[str, ModalEncoderModule]`):
                A dictionary of modal key and modal module.
                The modal module should be an instance of `ModalEncoderModule`.
            language_model (`PreTrainedModel`, *optional*):
                A language model to be used as a decoder.
                If not given, the model will be trained as an encoder-only model.
            init_projector_type (`str`, *optional*, defaults to `linear`):
                The type of projector layer to be initialized.
                If some encoder does not have a projector, it will be created
                using this argument.
                Supported types are `linear`, `mlp`, and `qformer`.
            init_activation (`str`, *optional*, defaults to `gelu`):
                The activation function when creating a projector layer.

        Examples:
        - An example of creating a VLM with CLIP vision encoder and llama-3
            ```
            from transformers.models.clip.modeling_clip import CLIPVisionModel
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            from cornstarch.models.multimodal_language_model import MultimodalModel, ModalEncoderModule

            vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_module = ModalEncoderModule(vision_encoder)

            language_model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

            vision_module.train()
            language_model.train()

            mm = MultimodalModel({"vision": vision_module}, language_model=language_model)
            ```

        - An example of using peft to fine-tune the pretrained models
            ```
            from cornstarch.models.multimodal_language_model import MultimodalModel, ModalEncoderModule

            from transformers.models.clip.modeling_clip import CLIPVisionModel
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            from accelerate import init_empty_weights
            from peft import get_peft_model, LoraConfig, TaskType

            with init_empty_weights():
                vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
                peft_config = LoraConfig(task_type=None, inference_mode=False, target_modules="all-linear")
                vision_encoder = get_peft_model(vision_encoder, peft_config)
                vision_module = ModalEncoderModule(vision_encoder)

                language_model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False)
                language_model = get_peft_model(language_model, peft_config)

            mm = MultimodalModel({"vision": vision_module}, language_model=language_model)
            ```
        """
        super().__init__()

        self.encoders = encoders
        self.encoders_args: dict[str, list[str]] = {}

        for modal_key, modal_module in encoders.items():
            if not isinstance(modal_module, ModalEncoderModule):
                raise ValueError(
                    f"Value of {modal_key} encoder should be an instance of ModalEncoderModule."
                )

            if language_model is not None:
                if modal_module.projector is None:
                    warnings.warn(
                        f"A projector for {modal_key} encoder is not given, "
                        "while it is required in multimodal with a language model. "
                        f"Creating a {init_projector_type} projector layer for the encoder. "
                        "If you want to load a pretrained projector, "
                        "please explicitly specify a projector in `ModalEncoderModule`."
                    )
                    projector_config = MultimodalProjectorConfig(
                        encoder_config=modal_module.module.config,
                        text_config=language_model.config,
                        projection_type=init_projector_type,
                        activation=init_activation,
                    )
                    modal_module.projector = MultimodalProjector(projector_config).to(
                        modal_module.module.device
                    )
                    modal_module.config = (modal_module.module.config, projector_config)

                # Check if the projector is compatible with the encoder and the language model
                projector_config: MultimodalProjectorConfig = (
                    modal_module.projector.config
                )
                if (
                    projector_config.in_features
                    != modal_module.module.config.hidden_size
                    or projector_config.out_features
                    != language_model.config.hidden_size
                    or projector_config.encoder_model_type
                    != modal_module.module.config.model_type
                    or projector_config.language_model_type
                    != language_model.config.model_type
                ):
                    raise ValueError(
                        f"Projector configuration for {modal_key} encoder is incompatible "
                        "to the current configuration: "
                        f"in_features (expected: {modal_module.module.config.hidden_size}, got: {projector_config.in_features}), "
                        f"out_features (expected: {language_model.config.hidden_size}, got: {projector_config.out_features}), "
                        f"encoder_model_type (expected: {modal_module.module.config.model_type}, got: {projector_config.encoder_model_type=}), "
                        f"language_model_type (expected: {language_model.config.model_type}, got: {projector_config.language_model_type=})."
                    )

            self.add_module(f"{modal_key}_encoder", modal_module)
            self.encoders_args[modal_key] = list(
                inspect.signature(modal_module.module.forward).parameters.keys()
            )

        self.language_model = language_model
        self.add_module("language_model", language_model)

    @classmethod
    def from_pretrained_multimodal_model(
        cls: MultimodalModel, pretrained_model_id: str, *args, **kwargs
    ) -> MultimodalModel:
        """
        Instantiate a cornstarch model from a pretrained multimodal model.

        Args:
            pretrained_model_id (`str`):
                A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
            args and kwargs are passed to from_pretrained().

        Currently supporting:
            llava-hf/llava-v1.5
            llava-hf/llava-v1.6
        """

        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained_model_id)

        if config.model_type == "llava":
            return LlavaModel(config).from_pretrained(*args, **kwargs)
        elif config.model_type == "llava_next":
            return LlavaNextModel(config).from_pretrained(*args, **kwargs)
        elif config.model_type == "internvl_chat":
            return InternVL2Model(config).from_pretrained(*args, **kwargs)
        elif config.model_type == "qwen2_vl":
            return Qwen2VLModel(config).from_pretrained(*args, **kwargs)
        else:
            raise NotImplementedError

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        for encoder in self.encoders.values():
            encoder.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            encoder.projector.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs
            )

        if self.language_model is not None:
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs
            )

    def train(self, mode: bool = True) -> MultimodalModel:
        for encoder in self.encoders.values():
            encoder.train(mode)

        if self.language_model is not None:
            self.language_model.train(mode)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Padding will be ignored by default should you provide it.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range `[0, config.n_positions - 1]`.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            use_cache (`bool`):
                If set to `True`, `past_key_values` key value states are returned and
                can be used to speed up decoding (see `past_key_values`).
            output_attentions (`bool`):
                Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under returned tensors for more detail.
            output_hidden_states (`bool`):
                Whether or not to return the hidden states of all layers.
                See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

            Inputs for modalities are passed as kwargs.
        """
        if self.language_model is None:
            # Does not support CLIP-like encoder only multimodal model yet
            raise NotImplementedError

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.language_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.language_model.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.language_model.config.return_dict
        )

        # step 1. forward the modal inputs to the encoders,
        # to get encoder embeddings of shape (batch_size, seq_len, hidden_size)
        encoders_inputs = {}
        encoders_outputs = {}
        for modal_key in self.encoders.keys():
            encoder_module: ModalEncoderModule = getattr(self, f"{modal_key}_encoder")
            args = {
                arg: kwargs[arg]
                for arg in self.encoders_args[modal_key]
                if arg in kwargs
            }

            for additional_arg in encoder_module.additional_args:
                if additional_arg in kwargs:
                    args[additional_arg] = kwargs[additional_arg]

            encoders_inputs[modal_key] = args

            encoders_outputs[modal_key] = encoder_module(
                **args,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # step 2. merge encoded multimodal features into text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        for modal_key in reversed(self.encoders.keys()):
            encoder_module: ModalEncoderModule = getattr(self, f"{modal_key}_encoder")
            inputs_embeds, attention_mask, position_ids, labels = (
                encoder_module.postprocess_projector_callback(
                    inputs=encoders_inputs[modal_key],
                    output=encoders_outputs[modal_key],
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            )

        language_model_inputs = dict(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # remove inputs that the language model doesn't accept
        language_model_arguments = list(
            inspect.signature(self.language_model.forward).parameters.keys()
        )
        for key in list(language_model_inputs.keys()):
            if key not in language_model_arguments:
                language_model_inputs.pop(key)

        return self.language_model(**language_model_inputs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_input_ids: Optional[bool] = True,
        **kwargs,
    ):
        """
        Generates sequences of token ids for models with a language modeling head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Padding will be ignored by default should you provide it.
        """

        if self.language_model is None:
            # Does not support CLIP-like encoder only multimodal model yet
            raise NotImplementedError

        # Filter out unused arguments
        language_model_params = list(
            inspect.signature(self.language_model.forward).parameters.keys()
        )

        encoders_inputs = {}
        encoders_outputs = {}
        for modal_key in self.encoders.keys():
            encoder_module: ModalEncoderModule = getattr(self, f"{modal_key}_encoder")
            args = {
                arg: kwargs[arg]
                for arg in self.encoders_args[modal_key]
                if arg in kwargs
            }

            for additional_arg in encoder_module.additional_args:
                if additional_arg in kwargs:
                    args[additional_arg] = kwargs[additional_arg]

            if encoder_module.module.main_input_name is not None:
                kwargs.pop(encoder_module.module.main_input_name, None)

            for arg in args:
                if arg not in language_model_params:
                    kwargs.pop(arg, None)

            encoders_inputs[modal_key] = args

            output_attentions = (
                output_attentions
                if output_attentions is not None
                else encoder_module.config[0].output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else encoder_module.config[0].output_hidden_states
            )

            encoders_outputs[modal_key] = encoder_module(
                **args,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            output_attentions = None
            output_hidden_states = None

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        for modal_key in reversed(self.encoders.keys()):
            encoder_module: ModalEncoderModule = getattr(self, f"{modal_key}_encoder")
            inputs_embeds, attention_mask, _, _ = (
                encoder_module.postprocess_projector_callback(
                    inputs=encoders_inputs[modal_key],
                    output=encoders_outputs[modal_key],
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=None,
                )
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.language_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.language_model.config.output_hidden_states
        )

        if not add_input_ids:
            input_ids = None

        return self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
