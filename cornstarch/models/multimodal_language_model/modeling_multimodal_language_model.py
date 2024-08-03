from __future__ import annotations

import inspect
import warnings
from collections import namedtuple
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from transformers.activations import get_activation
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llava import LlavaConfig, LlavaForConditionalGeneration
from transformers.models.llava_next import (
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
)

from cornstarch.models.multimodal_language_model import MultimodalProjectorConfig


class LlavaCallbacks:
    """A set of callbacks for Llava pretrained models.
    This is only for Llava <= 1.5, not compatible with Llava 1.6 (Llava-Next)"""

    def __init__(self, config: LlavaConfig):
        self.config = config

    def postprocess_vision_callback(
        self, output: BaseModelOutput | tuple
    ) -> BaseModelOutput | tuple:
        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy

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

    def postprocess_projector_callback(
        self,
        output: BaseModelOutput | tuple,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int = -1,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        FakeLlavaClass = namedtuple(
            "LlavaForConditionalGeneration", ["pad_token_id", "config"]
        )
        fake_llava = FakeLlavaClass(pad_token_id, self.config)
        inputs_embeds, attention_mask, labels, position_ids = (
            LlavaForConditionalGeneration._merge_input_ids_with_image_features(
                fake_llava, output[0], inputs_embeds, input_ids, attention_mask, labels
            )
        )

        return inputs_embeds, attention_mask, position_ids, labels


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

    def __init__(self, config: MultimodalProjectorConfig):
        super().__init__(config)
        self.gradient_checkpointing = False

        if config.projection_type == "linear":
            self.projection = nn.Linear(
                in_features=config.in_features,
                out_features=config.out_features,
            )
        elif config.projection_type == "mlp":
            self.in_proj = nn.Linear(
                in_features=config.in_features,
                out_features=config.out_features,
            )
            self.activation = get_activation(config.activation)
            self.out_proj = nn.Linear(
                in_features=config.out_features,
                out_features=config.out_features,
            )
        elif config.projection_type == "qformer":
            raise NotImplementedError

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, inputs_embeds: torch.Tensor, return_dict: bool = True
    ) -> Union[ModelOutput, tuple]:
        if self.gradient_checkpointing and self.training:
            if self.config.projection_type == "linear":
                outputs = self._gradient_checkpointing_func(
                    self.projection.__call__, inputs_embeds
                )
            elif self.config.projection_type == "mlp":
                outputs = self._gradient_checkpointing_func(
                    self.in_proj.__call__, inputs_embeds
                )
                outputs = self.activation(outputs)
                outputs = self._gradient_checkpointing_func(
                    self.out_proj.__call__, outputs
                )
            else:
                raise NotImplementedError
        else:
            if self.config.projection_type == "linear":
                outputs = self.projection(inputs_embeds)
            elif self.config.projection_type == "mlp":
                outputs = self.in_proj(inputs_embeds)
                outputs = self.activation(outputs)
                outputs = self.out_proj(outputs)
            else:
                raise NotImplementedError

        if not return_dict:
            return tuple(
                outputs,
            )

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

    def train(self, mode: bool = True) -> ModalModuleBase:
        self.module.train(mode)
        self.projector.train(mode)
        return self

    def get_modules(self) -> list[nn.Module]:
        modules = [self.module]
        if self.projector is not None:
            modules.append(self.projector)
        return modules


class ModalEncoderModule(ModalModuleBase):
    @staticmethod
    def prepend_modal_output_to_inputs_embeds(
        output: BaseModelOutput | tuple,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int = -1,
        ignore_ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simple postprocess_projector_callback that prepends the output of the `ModalEncoderModule` to the `inputs_embeds`."""
        # output[0] == output.last_hidden_state
        batch_size, num_tokens, _ = output[0].shape

        new_attention_mask = (
            torch.cat([torch.zeros((batch_size, num_tokens)), attention_mask], dim=1),
        )

        return (
            torch.cat(
                [torch.full((batch_size, num_tokens), pad_token_id), input_ids], dim=1
            ),
            torch.cat([output[0], inputs_embeds], dim=1),
            new_attention_mask,
            torch.sum(new_attention_mask, dim=1).unsqueeze(-1) - 1,
            torch.cat(
                [torch.full((batch_size, num_tokens), ignore_ignore_index), labels],
                dim=1,
            ),
        )

    def __init__(
        self,
        model: PreTrainedModel,
        projector: Optional[MultimodalProjector] = None,
        preprocess_callback: Callable[[Any], Any] = lambda x: x,
        postprocess_module_callback: Callable[
            [dict, BaseModelOutput | tuple], BaseModelOutput | tuple
        ] = lambda x: x,
        postprocess_projector_callback: Callable[
            [
                dict,
                BaseModelOutput | tuple,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
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
            preprocess_callback (`Callable[[Any], Any]`, *optional*):
                A function to preprocess inputs.
                Called before the encoder module is called to manipulate the inputs. Default is an identity function.
            postprocess_module_callback (`Callable[[dict, BaseModelOutput | tuple], BaseModelOutput]`, *optional*):
                A function to postprocess the output of the encoder module.
                Called after the encoder module is called and before the projector is called. Default is an identity function.
                Inputs to the callback:
                    dict: Inputs to the encoder module.
                    BaseModelOutput | tuple: Output of the encoder module.
            postprocess_projector_callback (`Callable[[dict, BaseModelOutput | tuple, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]`, *optional*):
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
                    int: `pad_token_id` from the language model backbone.
                    int: `ignore_index` from the language model backbone.
                Outputs from the callback:
                    torch.Tensor: new `inputs_embeds` to be used in the language model backbone.
                    torch.Tensor: new `attention_mask` to be used in the language model backbone.
                    torch.Tensor: new `position_ids` to be used in the language model backbone.
                    torch.Tensor: new `labels` to be used in the language model backbone.
        """
        super().__init__(model, projector)
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

        # Extract main input and call preprocess callback
        main_input_name = self.module.main_input_name
        kwargs[main_input_name] = self.preprocess_callback(kwargs[main_input_name])

        outputs: BaseModelOutput = self.module(
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        outputs = self.postprocess_module_callback(kwargs, outputs)

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

        if "llava-1.5" in pretrained_model_id:
            pretrained_model = LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_id, *args, **kwargs
            )
            pretrained_model.vision_tower.config.output_hidden_states = True
        elif "llava-v1.6" in pretrained_model_id:
            pretrained_model = LlavaNextForConditionalGeneration.from_pretrained(
                pretrained_model_id, *args, **kwargs
            )
            pretrained_model.vision_tower.config.output_hidden_states = True
        else:
            raise NotImplementedError

        vision_encoder = pretrained_model.vision_tower
        language_model = pretrained_model.language_model
        language_model.config.pad_token_id = pretrained_model.config.pad_token_id

        # Create projector
        pretrained_proj = pretrained_model.multi_modal_projector
        pretrained_proj_state_dict = pretrained_proj.state_dict()
        for key in pretrained_proj.state_dict().keys():
            pretrained_proj_state_dict[
                key.replace("linear_1", "in_proj").replace("linear_2", "out_proj")
            ] = pretrained_proj_state_dict.pop(key)

        projector_config = MultimodalProjectorConfig(
            encoder_config=vision_encoder.config,
            text_config=language_model.config,
            projection_type="mlp",
        )
        vision_projector = MultimodalProjector(projector_config)
        vision_projector.load_state_dict(pretrained_proj_state_dict, assign=True)

        llava_callbacks = LlavaCallbacks(pretrained_model.config)
        vision_tower = ModalEncoderModule(
            model=vision_encoder,
            projector=vision_projector,
            postprocess_module_callback=llava_callbacks.postprocess_vision_callback,
            postprocess_projector_callback=llava_callbacks.postprocess_projector_callback,
        )

        return cls(
            encoders={"vision": vision_tower},
            language_model=language_model,
        )

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
        encoders_outputs = {}
        for modal_key in self.encoders.keys():
            encoder_module = getattr(self, f"{modal_key}_encoder")
            args = {
                arg: kwargs[arg]
                for arg in self.encoders_args[modal_key]
                if arg in kwargs
            }
            if "output_attentions" in self.encoders_args[modal_key]:
                args["output_attentions"] = output_attentions
            if "output_hidden_states" in self.encoders_args[modal_key]:
                args["output_hidden_states"] = output_hidden_states
            if "return_dict" in self.encoders_args[modal_key]:
                args["return_dict"] = return_dict

            encoders_outputs[modal_key] = encoder_module(**args)

        encoders_outputs = torch.cat(encoders_outputs, dim=1)
        encoders_attention_mask = torch.ones(
            encoders_outputs.size()[:-1],
            dtype=torch.long,
            device=encoders_outputs.device,
        )

        # step 2. merge encoded multimodal features into text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        for modal_key in reversed(self.encoders.keys()):
            encoder_module: ModalEncoderModule = getattr(self, f"{modal_key}_encoder")
            inputs_embeds = encoder_module.postprocess_projector_callback(
                encoders_outputs[modal_key], inputs_embeds
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(
                encoders_attention_mask.device
            )
        attention_mask = torch.cat([encoders_attention_mask, attention_mask], dim=1)

        # Pad the labels to match the shape with LM input
        # Lables are padded by -100 (a default ignore_index in loss calculation)
        # so that they are ignored when calculating a loss.
        if labels is not None and labels.shape[1] != inputs_embeds.shape[1]:
            batch_size, seq_length = inputs_embeds.shape[:2]
            new_labels = torch.full((batch_size, seq_length), -100).to(labels.device)
            new_labels[:, -labels.shape[1] :] = labels
            labels = new_labels

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

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        encoders_inputs = {}
        encoders_outputs = {}
        for modal_key in self.encoders.keys():
            encoder_module: ModalEncoderModule = getattr(self, f"{modal_key}_encoder")
            args = {
                arg: kwargs[arg]
                for arg in self.encoders_args[modal_key]
                if arg in kwargs
            }
            for arg in args:
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
                    encoders_inputs[modal_key],
                    encoders_outputs[modal_key],
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    labels=None,
                    pad_token_id=self.language_model.config.pad_token_id,
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

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
