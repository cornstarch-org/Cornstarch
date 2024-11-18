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
from transformers.utils import logging

from cornstarch.models.internvl2.modeling_internvl_chat import (
    InternVLChatConfig,
    InternVLChatModel,
)
from cornstarch.models.multimodal_language_model import MultimodalProjectorConfig

logger = logging.get_logger(__name__)


def inject_modal_outputs_to_inputs_embeds(
    encoder_inputs: dict[str, dict],
    encoder_outputs: dict[str, BaseModelOutput | tuple],
    token_ids: dict[str, int],
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Based on token_ids, injects the outputs of the encoders to the input embeds.
    inputs_embeds must already have placeholders for the encoder outputs; unlike prepend_modal_output_to_inputs_embeds,
    which simply prepend the outputs to the inputs_embeds so that
    inputs_embeds does not need to have placeholders for the encoder outputs.

    It supports two types of attention_mask:
    - if 2D attention mask is given, it is used as is.
      this will be updated as a causal mask in model forward
    - if nothing is given, it creates a bit attention mask where
      attention information is stored in bits.
    """
    assert encoder_inputs.keys() == encoder_outputs.keys(), (
        "The keys of encoder_inputs and encoder_outputs must be the same. "
        f"Got {encoder_inputs.keys()} and {encoder_outputs.keys()}."
    )

    bit_attention_mask = torch.zeros(
        inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
    )

    for index, (modal_name, key) in enumerate(token_ids.items()):
        # output[0] == output.last_hidden_state
        modal_embeds = encoder_outputs[modal_name][0]
        modal_mask = input_ids == key
        inputs_embeds = inputs_embeds.masked_scatter(
            modal_mask.unsqueeze(-1).expand_as(inputs_embeds), modal_embeds
        )

        # set bit attention_mask
        bit_attention_mask |= modal_mask * (1 << index)

    # 0s in the bit_attention_mask at this moment are the positions of llm tokens
    llm_mask = bit_attention_mask == 0
    bit_attention_mask = bit_attention_mask.masked_fill(
        llm_mask, (1 << 62) | ((1 << len(token_ids)) - 1)
    )

    if attention_mask is None:
        # Create bit attention mask
        attention_mask = bit_attention_mask
        setattr(attention_mask, "cornstarch_is_bitattention", True)
        setattr(attention_mask, "cornstarch_num_encoders", len(encoder_outputs))
    else:
        assert attention_mask.ndim in [2, 3], (
            "The attention mask must have 2 or 3 dimensions. "
            f"Got {attention_mask.ndim} dimensions."
        )
        setattr(attention_mask, "cornstarch_is_bitattention", False)

    position_ids = (
        torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        .unsqueeze(0)
        .expand(inputs_embeds.shape[:2])
    )

    if labels is not None and (labels.shape[1] == input_ids.shape[1]):
        # labels are not padded
        new_labels = torch.full(
            inputs_embeds.shape[:2], ignore_index, device=labels.device
        )
        llm_labels = torch.masked_select(labels, llm_mask)
        new_labels = new_labels.masked_scatter(llm_mask, llm_labels)
        labels = new_labels

    return inputs_embeds, attention_mask, position_ids, labels


def prepend_modal_output_to_inputs_embeds(
    encoder_inputs: dict[str, dict],
    encoder_outputs: dict[str, BaseModelOutput | tuple],
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert encoder_inputs.keys() == encoder_outputs.keys(), (
        "The keys of encoder_inputs and encoder_outputs must be the same. "
        f"Got {encoder_inputs.keys()} and {encoder_outputs.keys()}."
    )

    # Prepend all values of encoder_outputs to inputs_embeds
    # output[0] == output.last_hidden_state
    all_modal_embeds = [
        encoder_outputs[modal_name][0] for modal_name in encoder_outputs.keys()
    ]
    all_modal_embeds.append(inputs_embeds)
    inputs_embeds = torch.cat(all_modal_embeds, dim=1)

    if attention_mask is None:
        num_modal_outputs = len(all_modal_embeds) - 1
        # Prefill text attention mask with (1 << 62 | sum(1 << i for i in range(num_modal_outputs)))
        bit_attention_mask = torch.full(
            input_ids.shape,
            (1 << 62) | ((1 << num_modal_outputs) - 1),
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        per_encoder_bit_attention_mask: list[torch.Tensor] = []
        # Prepend encoder bit attention mask
        for index, modal_output in enumerate(encoder_outputs.values()):
            bit_value = 1 << index
            modal_mask = torch.full(
                modal_output[0].shape[:2],
                bit_value,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            per_encoder_bit_attention_mask.append(modal_mask)

        bit_attention_mask = torch.cat(
            per_encoder_bit_attention_mask + [bit_attention_mask], dim=1
        )

        attention_mask = bit_attention_mask
        setattr(attention_mask, "cornstarch_is_bitattention", True)
        setattr(attention_mask, "cornstarch_num_encoders", len(encoder_outputs))
    else:
        if attention_mask.ndim == 2:
            attention_mask = torch.ones_like(
                inputs_embeds[:, :, 0], dtype=torch.long, device=inputs_embeds.device
            )
        else:
            pass

        setattr(attention_mask, "cornstarch_is_bitattention", False)

    position_ids = (
        torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        .unsqueeze(0)
        .expand(inputs_embeds.shape[:2])
    )

    if labels is not None and (labels.shape[1] == input_ids.shape[1]):
        # labels are not padded
        new_labels = torch.full(
            inputs_embeds.shape[:2], ignore_index, device=labels.device
        )
        new_labels[:, -labels.shape[1] :] = labels
        labels = new_labels

    return inputs_embeds, attention_mask, position_ids, labels


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
        )

        mm_model = MultimodalModel(
            encoders={"vision": vision_tower},
            language_model=language_model,
        )
        mm_model.set_token_ids({"vision": model.config.image_token_index})
        return mm_model

    @staticmethod
    def postprocess_vision_callback(
        inputs: dict,
        output: BaseModelOutput | tuple,
        model: LlavaForConditionalGeneration,
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
                self.postprocess_vision_callback,
                vision_feature_layer=self.config.vision_feature_layer,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            ),
            postprocess_projector_callback=functools.partial(
                self.postprocess_projector_callback,
                model=model,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            ),
        )

        mm_model = MultimodalModel(
            encoders={"vision": vision_tower},
            language_model=language_model,
        )
        mm_model.set_token_ids({"vision": model.config.image_token_index})
        mm_model.image_newline = model.image_newline
        return mm_model

    @staticmethod
    def preprocess_vision_callback(
        inputs: dict[str, Any], model: LlavaNextForConditionalGeneration
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
        inputs: dict,
        output: BaseModelOutput | tuple,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ) -> BaseModelOutput | tuple:
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
        inputs: dict,
        output: BaseModelOutput | tuple,
        model: LlavaNextForConditionalGeneration,
        vision_feature_select_strategy: Optional[str] = None,
    ) -> BaseModelOutput | tuple:
        pixel_values = inputs.get("pixel_values", None)

        if pixel_values is not None:
            image_sizes = inputs["image_sizes"]
            image_num_patches = [
                image_size_to_num_patches(
                    image_size=imsize,
                    grid_pinpoints=model.config.image_grid_pinpoints,
                    patch_size=model.config.vision_config.image_size,
                )
                for imsize in image_sizes
            ]

            # output[0] == output.last_hidden_state
            image_features = output[0]
            image_features = torch.split(image_features, image_num_patches, dim=0)

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            image_features, _ = model.pack_image_features(
                image_features,
                image_sizes,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=model.image_newline,
            )

            if isinstance(output, ModelOutput):
                output.last_hidden_state = image_features
            else:
                output = (image_features,) + output[1:]

        return output


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

        # TOOD (insujang): this doesn't work as expected,
        # since img_context_token_id is set during chat() or batch_chat().
        mm_model.set_token_ids({"vision": model.img_context_token_id})

        return mm_model

    @staticmethod
    def postprocess_vision_callback(
        inputs: dict,
        output: BaseModelOutput | tuple,
        model: InternVLChatModel,
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
            postprocess_module_callback=self.postprocess_vision_callback,
        )

        delattr(model, "visual")
        mm_model = MultimodalModel(
            encoders={"vision": vision_encoder},
            language_model=model,
        )
        mm_model.set_token_ids({"vision": model.config.image_token_id})

        return mm_model

    @staticmethod
    def preprocess_vision_callback(
        inputs: dict[str, Any], visual_dtype: torch.dtype
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

    @staticmethod
    def postprocess_vision_callback(
        inputs: dict,
        output: torch.Tensor,
    ) -> BaseModelOutput | tuple:
        if isinstance(output, torch.Tensor):
            return (output,)
        return output


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
            [dict, BaseModelOutput | tuple], BaseModelOutput | tuple
        ] = lambda inputs, output: output,
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
            postprocess_projector_callback (`Callable[[dict, BaseModelOutput | tuple], BaseModelOutput]`, *optional*):
                A function to postprocess the output of the projector layer.
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

        outputs: BaseModelOutput = self.module(
            # Filter out additional arguments
            **{k: v for k, v in kwargs.items() if k in module_params}
        )
        outputs = self.postprocess_module_callback(inputs=kwargs, output=outputs)

        if self.projector is None:
            return outputs

        outputs = self.projector(outputs[0], return_dict=return_dict)

        # Call postprocess projector callback
        return self.postprocess_projector_callback(inputs=kwargs, output=outputs)


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
        language_model: PreTrainedModel,
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
                ):
                    raise ValueError(
                        f"Projector configuration for {modal_key} encoder is incompatible "
                        "to the current configuration: "
                        f"in_features (expected: {modal_module.module.config.hidden_size}, got: {projector_config.in_features}), "
                        f"out_features (expected: {language_model.config.hidden_size}, got: {projector_config.out_features})."
                    )

            self.add_module(f"{modal_key}_encoder", modal_module)
            self.encoders_args[modal_key] = list(
                inspect.signature(modal_module.module.forward).parameters.keys()
            )

        self.language_model = language_model
        self.add_module("language_model", language_model)

        self.token_ids: dict[str, int] = None

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

    def set_token_ids(self, token_ids: dict[str, int]):
        self.token_ids = token_ids

    def merge_encoder_outputs(
        self,
        encoder_inputs: dict[str, dict],
        encoder_outputs: dict[str, BaseModelOutput | tuple],
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inject = self.token_ids is not None

        if attention_mask is not None:
            # 2d causal mask or full attention mask is used.
            assert attention_mask.ndim in [2, 3], (
                "Attention mask should be 2d (batch_size, seq_len) or 3d (batech_size, seq_len, seq_len), "
                f"got {attention_mask.ndim}."
            )

            total_seq_length = inputs_embeds.shape[1] + sum(
                output[0].shape[1] for output in encoder_outputs.values()
            )
            if attention_mask.ndim == 2:
                # check attention_mask shape is equal to input_ids shape
                assert attention_mask.shape == inputs_embeds.shape[
                    :2
                ] or attention_mask.shape == inputs_embeds.shape[:2] + (
                    total_seq_length,
                ), (
                    f"2d attention mask shape {attention_mask.shape} should be equal to either"
                    f"inputs_embeds shape {inputs_embeds.shape[:2]} or total_seq_length {total_seq_length}."
                )
            if attention_mask.ndim == 3:
                assert attention_mask.shape[1] == attention_mask.shape[2]
                # if full attention maks is given, it must include
                # the full attention mask "after" merging happens.

                if inject:
                    # No seq_length different expected.
                    assert (
                        attention_mask.shape[1] == inputs_embeds.shape[1]
                    ), f"3d attention mask seq_length {attention_mask.shape[1]} should be equal to inputs_embeds seq_length {inputs_embeds.shape[1]}."
                else:
                    # expect attention mask to already be prepended.
                    assert (
                        attention_mask.shape[1] == total_seq_length
                    ), f"3d attention mask seq_length {attention_mask.shape[1]} should be equal to total_seq_length {total_seq_length}."

        if inject:
            return inject_modal_outputs_to_inputs_embeds(
                encoder_inputs=encoder_inputs,
                encoder_outputs=encoder_outputs,
                token_ids=self.token_ids,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            return prepend_modal_output_to_inputs_embeds(
                encoder_inputs=encoder_inputs,
                encoder_outputs=encoder_outputs,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

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

            if "output_attentions" in self.encoders_args[modal_key]:
                args["output_attentions"] = output_attentions
            if "output_hidden_states" in self.encoders_args[modal_key]:
                args["output_hidden_states"] = output_hidden_states
            if "return_dict" in self.encoders_args[modal_key]:
                args["return_dict"] = return_dict

            encoders_inputs[modal_key] = args
            encoders_outputs[modal_key] = encoder_module(**args)

        # step 2. merge encoded multimodal features into text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # step 3. merge encoder outputs to llm inputs_embeds
        inputs_embeds, attention_mask, position_ids, labels = (
            self.merge_encoder_outputs(
                encoder_inputs=encoders_inputs,
                encoder_outputs=encoders_outputs,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        )

        # step 4. run llm with merged inputs_embeds
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

        inputs_embeds, attention_mask, *_ = self.merge_encoder_outputs(
            encoder_inputs=encoders_inputs,
            encoder_outputs=encoders_outputs,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,
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
