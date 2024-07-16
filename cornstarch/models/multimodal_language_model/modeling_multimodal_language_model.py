from __future__ import annotations

import inspect
import warnings
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.activations import get_activation
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers import LlavaForConditionalGeneration

from cornstarch.models.multimodal_language_model import MultimodalProjectorConfig


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

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
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

        return outputs


class ModalModuleType(Enum):
    Encoder = auto()
    Decoder = auto()


class ModalModule(nn.Module):
    """
    This is a wrapper of each modality model with a projector layer.

    Depending on the modal type, the execution order is different:
    - Encoder: input -> model -> projector -> output
    - Decoder: input -> projector -> model -> output
    """

    def __init__(
        self,
        model: PreTrainedModel,
        projector: Optional[MultimodalProjector] = None,
        modal_type: ModalModuleType = ModalModuleType.Encoder,
    ):
        super().__init__()
        self.module = model
        self.modal_type = modal_type
        self.projector = projector
        self.config = (model.config, projector.config if projector else None)

        if projector is not None:
            if modal_type == ModalModuleType.Encoder:
                assert projector.config.in_features == model.config.hidden_size, (
                    f"Input features of projector ({projector.config.in_features}) "
                    f"should be equal to hidden size of model ({model.config.hidden_size})."
                )
            else:
                assert projector.config.out_features == model.config.hidden_size, (
                    f"Output features of projector ({projector.config.out_features}) "
                    f"should be equal to hidden size of model ({model.config.hidden_size})."
                )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.projector is None:
            return self.module(*args, **kwargs)

        if self.modal_type == ModalModuleType.Encoder:
            return self.projector(self.module(*args, **kwargs)[0])
        else:
            return self.module(self.projector(*args, **kwargs))[0]

    def train(self, mode: bool = True) -> ModalModule:
        self.module.train(mode)
        self.projector.train(mode)
        return self

    def get_modules(self) -> list[nn.Module]:
        modules = [self.module]
        if self.projector is not None:
            modules.append(self.projector)
        return modules


class MultimodalModel(nn.Module):
    def __init__(
        self,
        encoders: dict[str, ModalModule],
        language_model: Optional[PreTrainedModel] = None,
        init_projector_type: str = "linear",
        init_activation: str = "gelu",
    ):
        """
        A representation of multimodal model, with arbitrary number of
        different types of encoders, and an optional large language model.

        Args:
            encoders (`dict[str, ModalModule]`):
                A dictionary of modal key and modal module.
                The modal module should be an instance of `ModalModule`.
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
            from cornstarch.models.multimodal_language_model import MultimodalModel, ModalModule

            vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_module = ModalModule(vision_encoder)

            language_model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

            vision_module.train()
            language_model.train()

            mm = MultimodalModel({"vision": vision_module}, language_model=language_model)
            ```

        - An example of using peft to fine-tune the pretrained models
            ```
            from cornstarch.models.multimodal_language_model import MultimodalModel, ModalModule

            from transformers.models.clip.modeling_clip import CLIPVisionModel
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            from accelerate import init_empty_weights
            from peft import get_peft_model, LoraConfig, TaskType

            with init_empty_weights():
                vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
                peft_config = LoraConfig(task_type=None, inference_mode=False, target_modules="all-linear")
                vision_encoder = get_peft_model(vision_encoder, peft_config)
                vision_module = ModalModule(vision_encoder)

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
            if not isinstance(modal_module, ModalModule):
                raise ValueError(
                    f"Value of {modal_key} encoder should be an instance of ModalModule."
                )

            if language_model is not None:
                if modal_module.projector is None:
                    warnings.warn(
                        f"A projector for {modal_key} encoder is not given, "
                        "while it is required in multimodal with a language model. "
                        f"Creating a {init_projector_type} projector layer for the encoder. "
                        "If you want to load a pretrained projector, "
                        "please explicitly specify a projector in `ModalModule`."
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

    def from_pretrained_multimodal_model(
        cls,
        model_id: str,
        encoders: dict[str, ModalModule],
        language_model: Optional[PreTrainedModel] = None,
    ) -> MultimodalModel:
        
        # Need comment

        if model_id == "llava-hf/llava-1.5-7b-hf":
            llava_model = LlavaForConditionalGeneration.from_pretrained(model_id, revision="main", torch_dtype="auto", device_map="cuda")
            
            # convert language model
            llava_state_dict = llava_model.language_model.state_dict()
            llava_state_dict_keys = llava_model.language_model.state_dict().keys()
            for key in llava_state_dict_keys:
                if llava_state_dict[key].shape[0] == 32064:
                    llava_state_dict[key] = llava_state_dict[key][:32000]
                if ".weight" in key:
                    llava_state_dict[key.replace(".weight", ".base_layer.weight")] = llava_state_dict[key]
            
            language_model.load_state_dict(llava_state_dict, strict=False)

            # create projector
            llava_proj = llava_model.multi_modal_projector
            llava_proj_state_dict = llava_proj.state_dict()
            for key in llava_proj.state_dict().keys():
                llava_proj_state_dict[key.replace('linear_1', 'in_proj'). replace('linear_2', 'out_proj')] = llava_proj_state_dict.pop(key)

            projector_config = MultimodalProjectorConfig(
                encoder_config=encoders["vision"].config,
                text_config=language_model.config,
                projection_type="mlp",
            )
            vision_projector = MultimodalProjector(projector_config)
            vision_projector.load_state_dict(llava_proj_state_dict)

            return MultimodalModel(
                encoders={"vision": ModalModule(model=encoders["vision"], projector=vision_projector)},
                language_model=language_model,
            )
        
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
        encoders_outputs = []
        for modal_key in self.encoders.keys():
            encoder_module = getattr(self, f"{modal_key}_encoder")
            args = {
                arg: kwargs[arg]
                for arg in self.encoders_args[modal_key]
                if arg in kwargs
            }
            encoders_outputs.append(encoder_module(**args))

        encoders_outputs = torch.cat(encoders_outputs, dim=1)
        encoders_attention_mask = torch.ones(
            encoders_outputs.size()[:-1],
            dtype=torch.long,
            device=encoders_outputs.device,
        )

        # step 2. merge encoded multimodal features into text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([encoders_outputs, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(
                encoders_attention_mask.device
            )
        attention_mask = torch.cat([encoders_attention_mask, attention_mask], dim=1)

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.language_model.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (
                logits,
                past_key_values,
                outputs.hidden_states,
                outputs.attentions,
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # Need commment 

        if self.language_model is None:
            # Does not support CLIP-like encoder only multimodal model yet
            raise NotImplementedError
        
        inputs_embeds = None
        
        encoders_outputs = []
        for modal_key in self.encoders.keys():
            encoder_module = getattr(self, f"{modal_key}_encoder")
            args = {
                arg: kwargs[arg]
                for arg in self.encoders_args[modal_key]
                if arg in kwargs
            }
            encoders_outputs.append(encoder_module(**args))

        encoders_outputs = torch.cat(encoders_outputs, dim=1)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([encoders_outputs, inputs_embeds], dim=1)

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            **kwargs
        )
