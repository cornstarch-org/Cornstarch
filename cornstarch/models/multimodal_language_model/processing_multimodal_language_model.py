import copy
import functools
import inspect
from typing import Callable, Union

import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers.utils import TensorType, logging

from cornstarch.models.multimodal_language_model.modeling_multimodal_language_model import (
    MultimodalModel,
)

logger = logging.get_logger(__name__)


def default_num_feature_calculation_func_audio_static(
    inputs: dict, outputs: dict, config: PretrainedConfig
) -> list[int]:
    num_features = config.max_source_positions
    return [num_features] * outputs["input_features"].shape[0]


def default_num_feature_calculation_func_vision_clip(
    inputs: dict, outputs: dict, config: PretrainedConfig
) -> list[int]:
    num_features = (config.image_size // config.patch_size) ** 2 + 1
    return [num_features] * outputs["pixel_values"].shape[0]


def default_num_feature_calculation_func_vision_static(
    inputs: dict, outputs: dict, config: PretrainedConfig
) -> list[int]:
    num_features = (config.image_size // config.patch_size) ** 2
    return [num_features] * outputs["pixel_values"].shape[0]


def default_num_feature_calculation_func_pixtral(
    inputs: dict, outputs: dict, config: PretrainedConfig
) -> list[int]:
    # output has "image_sizes", which has already been rescaled.
    # Use pixtral image processing functions to get the number of image tokens
    from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
    from transformers.models.pixtral.image_processing_pixtral import _num_image_tokens

    config: PixtralVisionConfig = config
    patch_size = (
        config.patch_size
        if isinstance(config.patch_size, (tuple, list))
        else (config.patch_size, config.patch_size)
    )

    num_image_tokens = []

    for image_size in outputs["image_sizes"]:
        num_tokens = np.prod(_num_image_tokens(image_size, patch_size))
        num_image_tokens.append(num_tokens)

    return num_image_tokens


def default_num_feature_calculation_func_qwen2vl(
    inputs: dict, outputs: dict, config: PretrainedConfig
) -> list[int]:
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

    config: Qwen2VLVisionConfig = config

    image_grid_thw = outputs.get("image_grid_thw", None)
    if image_grid_thw is None:
        return None

    merge_length = config.spatial_merge_size**2
    num_image_tokens = [
        image_grid_thw[i].prod() // merge_length for i in range(len(image_grid_thw))
    ]

    return num_image_tokens


processor_type_to_num_feature_calculation_func = {
    "ViTImageProcessor": default_num_feature_calculation_func_vision_static,
    "CLIPImageProcessor": default_num_feature_calculation_func_vision_clip,
    "SiglipImageProcessor": default_num_feature_calculation_func_vision_static,
    "BitImageProcessor": default_num_feature_calculation_func_vision_static,
    "PixtralImageProcessor": default_num_feature_calculation_func_pixtral,
    "Qwen2VLImageProcessor": default_num_feature_calculation_func_qwen2vl,
    "WhisperFeatureExtractor": default_num_feature_calculation_func_audio_static,
}


class MultimodalProcessor:
    """
    MultimodalModelProcessor is a class that processes text and images for multimodal language models.
    It is a composition of processors, feature extractors, and a tokenizer.

    Cornstarch MultimodalProcessor, different from the existing HuggingFace processors,
    takes inputs per modality and processes them separately to allow users to have more control over the processing.

    outputs: BatchFeature = mm_processor(encoder_inputs={
            "vision": {"images": ...},
            "audio": {"raw_speech": ...},
        }),
        llm_inputs={"text": ...},
    """

    def __init__(
        self,
        encoder_processors: dict[
            str, Union[BaseImageProcessor, SequenceFeatureExtractor]
        ],
        llm_tokenizer: PreTrainedTokenizer,
        model: MultimodalModel,
        num_feature_calculation_funcs: dict[
            str, Callable[[dict, dict], list[int] | list[list[int]]]
        ] = {},
        predefined_tokens: dict[str, str] = {},
    ):
        """
        MultimodalModelProcessor is a class that processes text, images, and any other multimodal inputs.
        Args:
            encoder_processors (dict[str, Union[BaseImageProcessor, SequenceFeatureExtractor]])
                A dictionary of modal_key to encoder processors. The model_key is the key used to identify the encoder.
                The encoder processor can be an image processor or a feature extractor.
            llm_tokenizer (PreTrainedTokenizer)
                The tokenizer used to tokenize the text inputs.
            num_feature_calculation_funcs (dict[str, Callable[[dict, dict], list[int | list[int]]])
                A dictionary of modal_key to a function that calculates the number of features for the encoder.
                When inputs are processed, the number of features is precalculated and
                corresponding modality tokens are added to the input.
                For this purpose, the processor needs to know how many modality tokens should be added.
                The callable function should take a dictionary of the modality encoder inputs
                and a dictionary of the modality encoder outputs,
                and return a list of the number of features (one per image for global batch),
                or a list of the list of the number of features (one per image for per batch).
            predefined_tokens (dict[str, str])
                A dictionary of modal_key to the token. This will override the default token for the corresponding modality encoders.
                By default the token for a modal_key is `<modal_key>`.
        """
        # Set the default num_feature_calculation_funcs
        for modal_key, processor in encoder_processors.items():
            processor_type = type(processor).__name__
            if processor_type in processor_type_to_num_feature_calculation_func:
                encoder = model.encoders[modal_key]
                num_feature_calculation_funcs[modal_key] = functools.partial(
                    processor_type_to_num_feature_calculation_func[processor_type],
                    config=encoder.module.config,
                )
            else:
                logger.warning(
                    f"num_feature_calculation_func for {modal_key} is not provided by Cornstarch."
                )

        self.encoder_processors = encoder_processors
        self.llm_tokenizer = llm_tokenizer
        self.num_feature_calculation_funcs = num_feature_calculation_funcs
        self.tokens: dict[str, str] = None
        self._set_modality_tokens(model, predefined_tokens)

        # check all the keys in the encoder_processors are in num_feature_calculation_funcs
        if set(encoder_processors.keys()) - set(num_feature_calculation_funcs.keys()):
            logger.warning_once(
                "The key in encoder_processors is not in num_feature_calculation_funcs.",
            )

    def _set_modality_tokens(
        self,
        model: MultimodalModel,
        predefined_tokens: dict[str, str],
    ):
        """
        Add the tokens as special tokens.

        By default the modality tokens are added as `<modal_key>`.
        If user wants to use different tokens, they can pass
        predefined_tokens as a dictionary of modal_key to the token.
        This will override the token for the corresponding modality encoders.
        """
        tokens = {
            modal_key: (
                predefined_tokens[modal_key]
                if modal_key in predefined_tokens
                else f"<{modal_key}>"
            )
            for modal_key in self.encoder_processors.keys()
        }

        self.llm_tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokens.values())}
        )
        token_ids = {
            modal_key: self.llm_tokenizer.convert_tokens_to_ids(token)
            for modal_key, token in tokens.items()
        }

        model.set_modality_token_ids(token_ids, len(self.llm_tokenizer))

        self.tokens = tokens

    def __call__(
        self,
        encoder_inputs: dict[str, dict] = {},
        llm_inputs: dict = {},
        return_tensors: str | TensorType = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        result: dict = {}

        if "text" not in llm_inputs:
            raise ValueError(
                "The llm_inputs should have a key 'text' for the text input."
            )

        text: list[str] = copy.deepcopy(llm_inputs["text"])
        if not isinstance(text, list):
            text = [text]

        for modal_key, encoder_input in encoder_inputs.items():
            if modal_key not in self.num_feature_calculation_funcs:
                raise ValueError(
                    f"num_feature_calculation_funcs for {modal_key} is not defined."
                )

            if modal_key not in self.tokens:
                raise ValueError(
                    f"tokens for {modal_key} is not defined. "
                    "Call MultimodalModel.set_modality_tokens() to set the tokens."
                )

            processor = self.encoder_processors[modal_key]

            # Filter kwargs for the processor
            processor_arguments = list(
                inspect.signature(processor.__call__).parameters.keys()
            )
            processor_inputs = {
                k: v for k, v in kwargs.items() if k in processor_arguments
            }
            processor_inputs.update(encoder_input)

            processor_result = processor(
                **processor_inputs, return_tensors=return_tensors
            )
            result.update(processor_result)

            num_features_list = self.num_feature_calculation_funcs[modal_key](
                processor_inputs, processor_result
            )

            # if num_features is a 2d array (list of num features per batch), flatten it.
            if isinstance(num_features_list[0], list):
                num_features_list = [
                    num_feature
                    for batch_num_features in num_features_list
                    for num_feature in batch_num_features
                ]

            index = 0
            for i in range(len(text)):
                while self.tokens[modal_key] in text[i]:
                    text[i] = text[i].replace(
                        self.tokens[modal_key],
                        "<|placeholder|>" * num_features_list[index],
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.tokens[modal_key])

        # Filter kwargs for the tokenizer
        tokenizer_arguments = list(
            inspect.signature(self.llm_tokenizer.__call__).parameters.keys()
        )
        tokenizer_inputs = {k: v for k, v in kwargs.items() if k in tokenizer_arguments}
        tokenizer_inputs.update(llm_inputs)
        tokenizer_inputs["text"] = text

        text_inputs = self.llm_tokenizer(
            **tokenizer_inputs, return_tensors=return_tensors
        )

        result.update(text_inputs)

        return BatchFeature(data={**result})

    def batch_decode(self, *args, **kwargs):
        return self.llm_tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.llm_tokenizer.decode(*args, **kwargs)

    def apply_chat_template(
        self, *args, **kwargs
    ) -> Union[str, list[int], list[str], list[list[int]], BatchEncoding]:
        return self.llm_tokenizer.apply_chat_template(*args, **kwargs)
