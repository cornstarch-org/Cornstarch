from typing import Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
)
from transformers.utils import TensorType


class MultimodalLanguageModelProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: BaseImageProcessor = None,
        tokenizer: PreTrainedTokenizerBase = None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
        images: ImageInput = None,
        return_tensors: str | TensorType = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You have to specify either text or images.")

        if images is not None:
            pixel_values = self.image_processor(
                images, return_tensors=return_tensors, **kwargs
            )["pixel_values"]
        else:
            pixel_values = None

        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
