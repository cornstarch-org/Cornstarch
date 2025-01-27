import copy
import inspect
from typing import Callable, Union

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


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
        num_feature_calculation_funcs: dict[str, Callable[[dict], int]] = {},
    ):
        """
        MultimodalModelProcessor is a class that processes text, images, and any other multimodal inputs.
        Args:
            encoder_processors (dict[str, Union[BaseImageProcessor, SequenceFeatureExtractor]])
                A dictionary of modal_key to encoder processors. The model_key is the key used to identify the encoder.
                The encoder processor can be an image processor or a feature extractor.
            llm_tokenizer (PreTrainedTokenizer)
                The tokenizer used to tokenize the text inputs.
            num_feature_calculation_funcs (dict[str, Callable[[dict], int]])
                A dictionary of modal_key to a function that calculates the number of features for the encoder.
                When inputs are processed, the number of features is precalculated and
                corresponding modality tokens are added to the input.
                For this purpose, the processor needs to know how many modality tokens should be added.
                The callable function should take a dictionary of the modality encoder inputs and return the number of features.
        """
        self.encoder_processors = encoder_processors
        self.llm_tokenizer = llm_tokenizer
        self.num_feature_calculation_funcs = num_feature_calculation_funcs
        self.tokens: dict[str, int] = None

        # check all the keys in the encoder_processors are in num_feature_calculation_funcs
        if set(encoder_processors.keys()) - set(num_feature_calculation_funcs.keys()):
            logger.warning_once(
                "The key in encoder_processors is not in num_feature_calculation_funcs.",
            )

    def set_modality_tokens(self, tokens: dict[str, str]) -> dict[str, int]:
        """
        Add the tokens as special tokens and return the corresponding token IDs.
        """
        self.tokens = tokens
        self.llm_tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokens.values())}
        )

        token_ids = {
            modal_key: self.llm_tokenizer.convert_tokens_to_ids(token)
            for modal_key, token in tokens.items()
        }

        return token_ids

    def __call__(
        self,
        encoder_inputs: dict[str, dict],
        llm_inputs: dict,
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

        num_features: dict[str, int] = {}
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

            num_features[modal_key] = self.num_feature_calculation_funcs[modal_key](
                processor_inputs
            )

            for i in range(len(text)):
                while self.tokens[modal_key] in text[i]:
                    text[i] = text[i].replace(
                        self.tokens[modal_key],
                        "<|placeholder|>" * num_features[modal_key],
                        1,
                    )
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
