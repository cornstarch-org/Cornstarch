from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
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
        padding_side: str,
        image_token_id: int,
        language_model: PreTrainedModel = None,
        vision_model: MultimodalEncoderProjector = None,
        **kwargs,
    ):
        r"""
        Args:
            config (`MultimodalLanguageModelConfig`):
                Configuration class for the model
            padding_side (`str`):
                The side to pad the image tokens. Either `left` or `right`.
                Get it from `MultimodalLanguageModelProcessor.tokenizer.padding_side`
            image_token_id (`int`):
                The image token id in the tokenizer
                Get it from `MultimodalLanguageModelProcessor.tokenizer.convert_tokens_to_ids("<image>")`
        """

        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")

        super().__init__(config)

        if language_model is None:
            language_model = AutoModelForCausalLM.from_config(
                config.text_config, **kwargs
            )

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

        self.padding_side = padding_side
        self.image_token_id = image_token_id

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def tie_weights(self):
        self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        feature_lens: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        image_token_id: int,
        labels: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        ignore_index: int = -100,
    ):
        """
        Merge input_ids with with image features into embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`)
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`)
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`)
                token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
                input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
                attention_mask for input_ids
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to None)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (7 patches)
                X, Y is on the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z s t u v _ _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ s t u v _ _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ _ o p q r Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ _ o p q r _ _ _ _ _ _ _ s t u v
                ]
        """
        with torch.no_grad():
            # ! in llava 1.6, number of patches is variable
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(
                    f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}"
                )
            batch_size, sequence_length = input_ids.shape

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_id
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens_batch = feature_lens.split(
                num_special_image_tokens.tolist(), dim=0
            )
            feature_lens_batch_sum = torch.tensor(
                [x.sum() for x in feature_lens_batch], device=feature_lens.device
            )
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1)
                - num_special_image_tokens
                + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where(
                (input_ids != image_token_id) & (attention_mask == 1)
            )
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_len_mask = special_image_token_mask.clone().long()
            special_image_len_mask[special_image_len_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_len_mask + 1), -1) - 1
            if padding_side == "left":
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        final_labels = None
        if labels is not None:
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(
                torch.long
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_image_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_image_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_image_indices
            ]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        with torch.no_grad():
            image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if padding_side == "left":
                # exclude padding on the left
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        return final_embedding, final_attention_mask, position_ids, final_labels

    def _pad_embeddings(self):
        raise NotImplementedError

    def pack_image_features(
        self,
        image_features: list[torch.Tensor],
        image_sizes: torch.LongTensor,
        num_patches: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.LongTensor` of shape `(num_images, 2)`)
                The size of the image (height, width) for each image in the batch.
            num_patches (`torch.LongTensor` of shape `(num_images, 2)`)
                The number of patches per image in the batch.

        Returns:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`)
            feature_lens (`torch.LongTensor` of shape `(num_images)`)
                feature length of each image in image_features
        """
        assert len(image_sizes) == len(num_patches)

        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = (
                    self.config.vision_config.image_size
                    // self.config.vision_config.patch_size
                )
                if height * width != base_image_feature.shape[0]:
                    raise ValueError(
                        "The number of patches is not consistent with the image size."
                    )

                num_patch_height, num_patch_width = num_patches[image_idx]
                image_feature = image_feature.view(
                    num_patch_height, num_patch_width, height, width, -1
                )
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(
            feature_lens, dtype=torch.long, device=image_features.device
        )
        return image_features, feature_lens

    def merge_image_features_with_input_ids(
        self,
        input_ids: torch.LongTensor,
        image_features: torch.FloatTensor,
        feature_lens: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        ignore_index: int = -100,
    ) -> tuple[
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
        Optional[torch.LongTensor],
    ]:
        r"""
        Merge image features with input_ids into embeddings

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
                input_ids of tokens, possibly filled with image token
            image_features (`torch.FloatTensor` of shape `(all_feature_lens, embed_dim)`)
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`)
                The length of visual embeddings of each image as stacked in `image_features`
            num_patches (`torch.LongTensor` of shape `(batch_size)`)
                The number of patches per image in the batch
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`)
                token embeddings before merging with visual embeddings
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
                attention_mask for input_ids
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            labels (`torch.LongTensosr` of shape `(batch_size, sequence_length)`, *optional*, defaults to None)

        Returns:
            A tuple of new_embedding, new_attention_mask, position_ids, new_labels
            new_embedding (`torch.FloatTensor` of shape `(batch_size, new_sequence_length, embed_dim)`)
                Merged embeddings of text and image features
            new_attention_mask (`torch.LongTensor` of shape `(batch_size, new_sequence_length)`)
                Merged attention mask of text and image features
            position_ids (`torch.LongTensor` of shape `(batch_size, new_sequence_length)`)
                Position ids of merged embeddings
            new_labels (`torch.LongTensor` of shape `(batch_size, new_sequence_length)`, *optional*)
                Labels recalculated to support training (if provided)

        Explanation:
            Each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (7 patches)
                X, Y is on the same sequence (in-context learning)

            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z s t u v _ _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ s t u v _ _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ _ o p q r Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ _ o p q r _ _ _ _ _ _ _ s t u v
                ]
        """
        image_token_id: int = self.image_token_id
        left_padding: bool = self.padding_side == "left"

        with torch.no_grad():
            num_images = feature_lens.size()
            num_image_features, embed_dim = image_features.shape
            batch_size, sequence_length = input_ids.shape

            if feature_lens.sum() != num_image_features:
                raise ValueError(
                    f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}"
                )

            # 1. Create a mask to know where special image tokens are
            # image_token_mask: [bsz, seqlen]
            image_token_mask: torch.BoolTensor = input_ids == image_token_id
            # num_special_image_tokens: [bsz]
            num_image_tokens = torch.sum(image_token_mask, dim=-1)
            # Reserve for padding of num_images
            total_num_image_tokens = torch.sum(image_token_mask)
            if total_num_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_image_tokens}) "
                    f"different from num_images ({num_images})."
                )

            # Compute the maximum embed dimension. The maximum embed dimension is the new sequence length
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens_batch: tuple[torch.Tensor, ...] = feature_lens.split(
                num_image_tokens.tolist(), dim=0
            )
            feature_lens_batch_sum: torch.Tensor = torch.tensor(
                [x.sum() for x in feature_lens_batch], device=feature_lens.device
            )
            embed_sequence_lengths: torch.Tensor = (
                (attention_mask == 1).long().sum(-1)
                - num_image_tokens
                + feature_lens_batch_sum
            )
            max_embed_dim: int = embed_sequence_lengths.max()

            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens im merged image-text sequence.
            # `image_token_mask` identifies image tokens.
            # Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            batch_indices, non_image_indices = torch.where(
                (input_ids != image_token_id) & (attention_mask == 1)
            )
            image_len_mask: torch.LongTensor = image_token_mask.clone().long()
            image_len_mask[image_len_mask == 1] = feature_lens - 1
            new_token_positions: torch.Tensor = (
                torch.cumsum((image_len_mask + 1), dim=-1) - 1
            )
            if left_padding:
                # shift right token positions so that they are ending at the same number
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite: torch.Tensor = new_token_positions[
                batch_indices, non_image_indices
            ]

        # Create full embedding, already padded to the maximum position
        new_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        new_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        new_labels = None
        if labels is not None:
            new_labels = torch.full_like(new_attention_mask, ignore_index).to(
                torch.long
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices = batch_indices.to(target_device)
        non_image_indices = non_image_indices.to(target_device)
        text_to_overwrite = text_to_overwrite.to(target_device)
        attention_mask = attention_mask.to(target_device)

        # Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        new_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_image_indices
        ]
        new_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_image_indices
        ]
        if labels is not None:
            new_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_image_indices
            ]

        # Fill the embeddings corresponding to the images.
        # Anything that is not `text_positions` needs filling
        with torch.no_grad():
            image_to_overwrite: torch.Tensor = torch.full(
                (batch_size, max_embed_dim),
                True,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices: torch.Tensor = (
                torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            )
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens: torch.Tensor = embed_sequence_lengths[:, None].to(
                target_device
            )

            if left_padding:
                # Exclude padding on the left
                val: torch.Tensor = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # Exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )

        new_embedding[image_to_overwrite] = (
            image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        new_attention_mask |= image_to_overwrite
        position_ids = (new_attention_mask.cumsum(dim=-1), -1).masked_fill_(
            (new_attention_mask == 0), 1
        )

        return new_embedding, new_attention_mask, position_ids, new_labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        num_patches: Optional[list[torch.LongTensor]] = None,
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
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`)
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using `tokenizer` encoding methods.
            pixel_values (`torch.FloatTensor` of shape either
                `(batch_size, num_patches, channels, height, width)` or
                `(total_num_patches, channels, height, width)`, *optional*, defaults to `None`)
                Pixel values for the image. Pixel values can be obtained using `ImageProcessor`.
            image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*, defaults to `None`)
                The size of the image (height, width) for each image in the batch.
                The image sizes are the original image sizes before they are split into patches.
            num_patches (`List[torch.LongTensor]` of length `batch_size`, *optional*, defaults to `None`)
                The number of patches per image in the batch.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`)
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests

            >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
            >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

            >>> # Generate
            >>> generate_ids = model.generate(**inputs, max_length=30)
            >>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
            ```
        """
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
            # In case image_token_id is not in the embeddings (extra token but embeddings don't have it)
            for_inputs_embeds_ids = input_ids.clone()
            for_inputs_embeds_ids[(input_ids == self.image_token_id)] = 0
            inputs_embeds = self.language_model.get_input_embeddings()(
                for_inputs_embeds_ids
            )

            # Merging encoded multimodal features into text embeddings only happens in the first token generation
            if past_key_values is None:
                # Encoder image into image features, and merge them into text embeddings
                if pixel_values is not None:
                    # pixel_values is not None but is empty -> text only cases
                    if pixel_values.size(0) == 0:
                        # there are no images
                        pass
                    else:
                        # figure out if pixel_values is concatenate or stacking
                        if pixel_values.dim() == 5:
                            # stacking: e.g. [2, 5, 3, 336, 336], convert to concat
                            # remove padded patches and concatenate them
                            _pixel_values_list = [
                                pix_val[:num_patch]
                                for pix_val, num_patch in zip(pixel_values, num_patches)
                            ]
                            pixel_values = torch.cat(_pixel_values_list, dim=0)
                        elif pixel_values.dim() != 4:
                            # concat: e.g. [8, 3, 336, 336]
                            raise ValueError(
                                f"pixel_values of shape {pixel_values.shape}, expected to be of 4 or 5 dimensions"
                            )

                        image_features = self.vision_model(pixel_values)
                        image_features = torch.split(image_features, num_patches, dim=0)

                        image_features, feature_lens = self.pack_image_features(
                            image_features,
                            image_sizes,
                            num_patches,
                        )

                        inputs_embeds, attention_mask, position_ids, labels = (
                            self.merge_image_features_with_input_ids(
                                input_ids=input_ids,
                                image_features=image_features,
                                image_sizes=image_sizes,
                                num_patches=num_patches,
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                labels=labels,
                            )
                        )
            # In case past_key_value is not None, we are in the case of generation with cache
            else:
                assert (
                    input_ids.shape[1] == 1
                ), f"Expected one unprocessed token, got {input_ids.shape[1]}"

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
        input_ids: torch.LongTensor,
        past_key_values: Cache | torch.FloatTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: torch.LongTensor = None,
        num_patches: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
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
            elif self.config.image_token_index in input_ids:
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
                "image_sizes": image_sizes,
                "num_patches": num_patches,
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

        tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path)
        tokenizer.add_tokens("<image>")
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

        model = cls(
            config=config,
            padding_side=tokenizer.padding_side,
            image_token_id=image_token_id,
            language_model=language_model,
            vision_model=vision_model,
        )

        return model
