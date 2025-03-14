from __future__ import annotations

import functools
import inspect
from types import MethodType
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from peft.peft_model import PeftModel
from transformers.activations import get_activation
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto.configuration_auto import AutoConfig
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

from cornstarch.kernel.interface import (
    bitfield_attention_forward,
    create_bitfield_attention_mask,
)
from cornstarch.models.multimodal_language_model import MultimodalProjectorConfig

logger = logging.get_logger(__name__)

ALL_ATTENTION_FUNCTIONS.update({"bitfield_attention": bitfield_attention_forward})


class LlavaNextModel:
    """
    A class for Llava-next pretrained models.
    """

    def __init__(self, config: LlavaNextConfig):
        self.config = config

    def from_pretrained(self, *args, **kwargs) -> MultimodalModel:
        model: LlavaNextForConditionalGeneration = (
            LlavaNextForConditionalGeneration.from_pretrained(
                self.config.name_or_path, config=self.config, *args, **kwargs
            )
        )
        attn_implementation = model.language_model.config._attn_implementation

        model.vision_tower.config.output_hidden_states = True
        vision_encoder = model.vision_tower
        language_model = model.language_model
        language_model.config.pad_token_id = (
            model.config.pad_token_id if model.config.pad_token_id is not None else -1
        )

        # Create a projector
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
        mm_model.set_modality_token_ids({"vision": model.config.image_token_index})
        mm_model.image_newline = model.image_newline
        mm_model.language_model.config._attn_implementation = attn_implementation
        return mm_model

    @staticmethod
    def preprocess_vision_callback(
        inputs: dict[str, Any], model: LlavaNextForConditionalGeneration
    ) -> dict[str, Any]:
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["image_sizes"]

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=model.config.image_grid_pinpoints,
                patch_size=model.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
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
        model: LlavaNextForConditionalGeneration,
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
        inputs: dict,
        output: BaseModelOutput | tuple,
        model: LlavaNextForConditionalGeneration,
    ) -> BaseModelOutput | tuple:
        pixel_values = inputs.get("pixel_values", None)
        vision_feature_select_strategy = model.config.vision_feature_select_strategy

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
                output = ModelOutput(hidden_states=image_features)
            else:
                output = (image_features,) + output[1:]

        return output


class Qwen2VLModel:
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
        attn_implementation = model.config._attn_implementation

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
        mm_model.set_modality_token_ids({"vision": model.config.image_token_id})
        mm_model.language_model.config._attn_implementation = attn_implementation

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
            if output.ndim == 2:
                # Add batch dimension here
                output.unsqueeze_(0)
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
        self, hidden_states: torch.Tensor, return_dict: bool = True
    ) -> Union[ModelOutput, tuple]:
        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(self.projection, hidden_states)
        else:
            outputs = self.projection(hidden_states)

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

    def train(self, module: bool = True, projector: bool = True) -> ModalModuleBase:
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

        # Call preprocess callback
        kwargs = self.preprocess_callback(inputs=kwargs)

        # Filter out arguments
        module_params = list(inspect.signature(self.module.forward).parameters.keys())
        module_params = {k: v for k, v in kwargs.items() if k in module_params}

        outputs: BaseModelOutput | tuple | torch.Tensor = self.module(**module_params)

        if isinstance(outputs, torch.Tensor):
            outputs = BaseModelOutput(last_hidden_state=outputs)
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
        preprocess_llm_callback: Optional[Callable] = None,
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
                    logger.warning(
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

        # FIXME: bitfield attention currently is not functional.
        # self.update_language_model_to_use_bitfield_attention_mask(language_model)
        self.language_model = language_model
        self.add_module("language_model", language_model)

        self.token_ids: dict[str, int] = None
        self.preprocess_llm_callback = preprocess_llm_callback

    def update_language_model_to_use_bitfield_attention_mask(
        self, language_model: PreTrainedModel
    ):
        if isinstance(language_model, PeftModel):
            language_model = language_model.get_base_model()

        language_model.prepare_inputs_for_generation = MethodType(
            functools.partial(
                MultimodalModel.prepare_inputs_for_generation_of_language_model,
                original_func=language_model.prepare_inputs_for_generation,
            ),
            language_model,
        )

        if not hasattr(language_model, "_update_causal_mask"):
            # TODO: does this always work?
            language_model = language_model.model

        assert hasattr(language_model, "_update_causal_mask"), (
            "The language model should have _update_causal_mask method. "
            "Please check if the language model is from HuggingFace Transformers."
        )

        language_model.config._attn_implementation = "bitfield_attention"
        language_model._update_causal_mask = MethodType(
            MultimodalModel.check_bitfield_attention_mask,
            language_model,
        )

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
        """

        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained_model_id)

        if config.model_type == "llava_next":
            return LlavaNextModel(config).from_pretrained(*args, **kwargs)
        elif config.model_type == "qwen2_vl":
            return Qwen2VLModel(config).from_pretrained(*args, **kwargs)
        else:
            raise NotImplementedError

    @staticmethod
    def check_bitfield_attention_mask(
        self, attention_mask: torch.Tensor, *args, **kwargs
    ):
        """
        A replacement function for Model._update_causal_mask() to avoid updating the attention mask.
        Plus, it checks if the attention mask is a bitfield attention mask.
        """
        if self.training:
            assert (
                attention_mask is not None
                and attention_mask.dtype == torch.int64
                and (attention_mask > 1).any()
            ), "The attention mask should be a bitfield attention mask."

        return attention_mask

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        if (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"] is True
        ):
            gradient_checkpointing_kwargs["use_reentrant"] = False

        for encoder in self.encoders.values():
            encoder.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            encoder.projector.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs
            )

        if self.language_model is not None:
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs
            )

    def train(
        self, encoders_mode: dict[str, tuple[bool, bool]] = None, llm_mode: bool = None
    ) -> MultimodalModel:
        """
        Set the training mode for the model components.

        Args:
            encoders_mode: dict[str, tuple[bool, bool]]
                A dictionary of encoder key -> (encoder_mode, projector_mode).
            llm_mode: bool
                The training mode for the language model.
        """
        super().train(True)

        if encoders_mode is None:
            if llm_mode is None:
                return

            encoders_mode = {
                encoder_key: (llm_mode, llm_mode) for encoder_key in self.encoders
            }

        for encoder_key, (encoder_mode, projector_mode) in encoders_mode.items():
            if encoder_key not in self.encoders:
                continue

            encoder = self.encoders[encoder_key]
            for p in encoder.module.parameters():
                p.requires_grad_(encoder_mode)
            for p in encoder.projector.parameters():
                p.requires_grad_(projector_mode)

        if self.language_model is not None and llm_mode is not None:
            # only change language model train mode when llm_mode is given to support peft
            for p in self.language_model.parameters():
                p.requires_grad_(llm_mode)

    def set_modality_token_ids(
        self, token_ids: dict[str, int], new_num_tokens: int = 0
    ):
        """
        Store modality token ids to the model, so that later
        it can replace the tokens with modality encoder outputs.

        MultimodalProcessor is responsible for calling this function.
        Do not call manually.
        """
        self.token_ids = token_ids

    def merge_encoder_outputs(
        self,
        encoders_outputs: dict[str, BaseModelOutput | tuple],
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Based on token_ids, merge the outputs of the encoders to the input embeds.
        inputs_embeds must already have placeholders for the encoder outputs, otherwise
        it will raise a ValueError.

        Arguments:
            encoders_inputs: dict[str, dict]
                A dictionary of encoder key -> corresponding encoder inputs.
            encoders_outputs: dict[str, BaseModelOutput | tuple]
                A dictionary of encoder key -> corresponding encoder outputs.
            input_ids: torch.Tensor
                The input for the LLM. This must have placeholders for the encoder outputs.
            inputs_embeds: torch.Tensor
                The input embeddings for the LLM. This can be generated by calling
                `llm.get_input_embeddings()(input_ids)`.
            attention_mask: torch.Tensor
                The original 2D attention mask for the LLM (batch_size, sequence_length).
                It should be binary that includes either 1 or 0.

        Returns:
            torch.Tensor: The merged input embeddings.
            torch.Tensor: A generated bitfield attention mask.
        """
        if self.token_ids is None:
            raise ValueError(
                "token_ids must be set before merging encoder outputs. "
                "Call `MultimodalModel.set_modality_tokens()` to set token ids."
            )

        sequence_lengths = attention_mask.to(dtype=torch.int64, device="cpu").sum(dim=1)
        for modal_key, output in encoders_outputs.items():
            output: torch.Tensor = output[0]

            if output.ndim > 2:
                # logger.warning_once(
                #     f"{modal_key} encoder output shape {output.shape} should be 2d (num_features, hidden_size). "
                #     "Add postprocess_projector_callback to the encoder module to convert the shape. "
                #     "For most encoders that Cornstarch support already has the shape conversion implementation; "
                #     "you can see how the conversion is done in the source code.",
                # )
                output = output.reshape(-1, output.shape[-1])
            if output.shape[-1] != inputs_embeds.shape[-1]:
                raise ValueError(
                    f"Expected encoder output hidden_size {output.shape[-1]} to be equal to inputs_embeds hidden_size {inputs_embeds.shape[-1]}."
                )

            if modal_key not in self.token_ids:
                raise ValueError(
                    f"Token ID for {modal_key} is not set. "
                    "Call `MultimodalModel.set_modality_tokens()` to set token ids."
                )

            token_id = self.token_ids[modal_key]
            num_modal_tokens = (input_ids == token_id).sum().item()
            num_modal_features = output.shape[0]
            if num_modal_tokens != num_modal_features:
                raise ValueError(
                    f"Number of {modal_key} tokens {num_modal_tokens} should be equal to number of {modal_key} features {num_modal_features}."
                )

            modal_mask = (
                (input_ids == token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(modal_mask, output)

            sequence_lengths = sequence_lengths + (input_ids.to("cpu") == token_id).sum(
                dim=1
            )

        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        for i, sequence_length in enumerate(sequence_lengths.tolist()):
            attention_mask[i, sequence_length:] = 0

        # attention_mask = create_bitfield_attention_mask(input_ids, self.token_ids)
        # for i, sequence_length in enumerate(sequence_lengths.tolist()):
        #     attention_mask[i, sequence_length:] = 0

        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
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

            if hasattr(encoder_module.module, "main_input_name"):
                # if the main input is not in args, this encoder should not be executed
                if encoder_module.module.main_input_name not in args:
                    continue

            if "output_attentions" in self.encoders_args[modal_key]:
                args["output_attentions"] = output_attentions
            if "output_hidden_states" in self.encoders_args[modal_key]:
                args["output_hidden_states"] = output_hidden_states
            if "return_dict" in self.encoders_args[modal_key]:
                args["return_dict"] = return_dict

            encoders_outputs[modal_key] = encoder_module(**args)

        # step 2. merge encoded multimodal features into text embeddings
        # mask out special tokens from input_ids to avoid out of index error
        # and use it as an input to embedding.
        token_mask = torch.isin(
            input_ids,
            torch.tensor(list(self.token_ids.values()), device=input_ids.device),
        )
        input_ids_masked = input_ids.clone()
        input_ids_masked[token_mask] = self.language_model.config.eos_token_id
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids_masked)

        labels_masked = labels.clone()
        labels_masked[token_mask] = -100

        # step 3. merge encoder outputs to llm inputs_embeds
        inputs_embeds, attention_mask = self.merge_encoder_outputs(
            encoders_outputs=encoders_outputs,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # step 4. run llm with merged inputs_embeds
        language_model_inputs = dict(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            hidden_states=None,
            labels=labels_masked,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        language_model_inputs.update(kwargs)

        if self.preprocess_llm_callback is not None:
            # filter out inputs that the preprocess_llm_callback doesn't accept
            callback_arguments = list(
                inspect.signature(self.preprocess_llm_callback).parameters.keys()
            )

            callback_inputs = {
                key: value
                for key, value in language_model_inputs.items()
                if key in callback_arguments
            }

            callback_outputs = self.preprocess_llm_callback(**callback_inputs)
            language_model_inputs.update(callback_outputs)

        # remove inputs that the language model doesn't accept
        language_model_arguments = list(
            inspect.signature(self.language_model.forward).parameters.keys()
        )
        if "kwargs" not in language_model_arguments:
            for key in list(language_model_inputs.keys()):
                if key not in language_model_arguments:
                    language_model_inputs.pop(key)

        return self.language_model(**language_model_inputs)

    def prepare_inputs_for_generation_of_language_model(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        original_func: Callable = None,
        **kwargs,
    ):
        """
        HF Transformers GenerationMixin.prepare_inputs_for_generation() generates
        `position_ids` on the fly based on the value of attention mask,
        which makes itself incompatible with bitfield attention mask.

        This function is to replace the language model's `prepare_inputs_for_generation`
        to avoid an error when using bitfield attention mask.
        """
        bool_attention_mask = (attention_mask != 0).long()

        model_inputs = original_func(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=bool_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if cache_position[0] == 0:
            # This is the prefill stage; replace the bool attention mask with bitfield attention mask
            model_inputs["attention_mask"] = attention_mask
        else:
            # In the decoding stage, Cornstarch no longer uses bitfield attention mask.
            # Don't have to take care about the value of attention mask.
            pass

        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Generates sequences of token ids for models with a language modeling head.

        Args:
            inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Padding will be ignored by default should you provide it.
        """

        if self.language_model is None:
            # Does not support CLIP-like encoder only multimodal model yet
            raise NotImplementedError

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

            if hasattr(encoder_module.module, "main_input_name"):
                # if the main input is not in args, this encoder should not be executed
                if encoder_module.module.main_input_name not in args:
                    continue

            for arg in args:
                kwargs.pop(arg, None)

            encoders_outputs[modal_key] = encoder_module(
                **args,
                output_attentions=encoder_module.config[0].output_attentions,
                output_hidden_states=encoder_module.config[0].output_hidden_states,
                return_dict=True,
            )

        token_mask = torch.isin(
            input_ids,
            torch.tensor(list(self.token_ids.values()), device=input_ids.device),
        )
        input_ids_masked = input_ids.clone()
        input_ids_masked[token_mask] = self.language_model.config.eos_token_id
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids_masked)

        inputs_embeds, attention_mask = self.merge_encoder_outputs(
            encoders_outputs=encoders_outputs,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # remove inputs that the language model doesn't accept
        language_model_inputs = dict(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        language_model_inputs.update(kwargs)

        return self.language_model.generate(**language_model_inputs)
