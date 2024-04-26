import math
from typing import Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
)
from transformers.utils import TensorType


# Copied from transformers/v4.40.0/src/transformers/image_processing_utils.py
def select_best_resolution(
    original_size: tuple[int, int], possible_resolutions: list[tuple[int, int]]
) -> tuple[int, int]:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit: tuple[int, int] = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = (
            int(original_width * scale),
            int(original_height * scale),
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


# Copied from transformers/a0102a425dc8d01fddf215444aa2e54dfd8b7eb2/src/transformers/models/llava_next/modeling_llava_next.py
def image_size_to_num_patches(image_size: tuple[int, int], patch_size: int) -> int:
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (height, width).
        patch_size (`int`):
            The size of each image patch.

    Returns:
        The number of image patchs.
    """
    height, width = image_size
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches


# Copied from transformers/a0102a425dc8d01fddf215444aa2e54dfd8b7eb2/src/transformers/models/llava_next/image_processing_llava_next.py
def _divide_to_patches(
    image: np.ndarray, patch_size: int, input_data_format: ChannelDimension
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array`):
            The input image.
        patch_size (`int`):
            The size of each patch.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        list: A list of np.array representing the patches.
        tuple: The number of patches in the image.
    """
    patches = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if input_data_format == ChannelDimension.LAST:
                patch = image[i : i + patch_size, j : j + patch_size]
            else:
                patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches, (height // patch_size, width // patch_size)


def _get_patch_output_size(
    image: np.ndarray,
    target_resolution: tuple[int, int],
    input_data_format: ChannelDimension,
) -> tuple[int, int]:
    original_height, original_width = get_image_size(
        image, channel_dim=input_data_format
    )
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


class ImageProcessorWrapper(BaseImageProcessor):
    """
    Inspired by LLaVA-NeXT, it implements additional techniques for processing
    high resolution images as explained in the [LLaVa paper](https://arxiv.org/abs/2310.03744).

    Copied from:
    https://github.com/huggingface/transformers/blob/a0102a425dc8d01fddf215444aa2e54dfd8b7eb2/src/transformers/models/llava_next/image_processing_llava_next.py
    """

    def __init__(self, image_processor: BaseImageProcessor):
        self.image_processor = image_processor
        shortest_edge_or_height = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else image_processor.size["height"]
        )
        self.crop_size = get_size_dict(shortest_edge_or_height, default_to_square=True)

        patch_size = self.crop_size["height"]
        self.image_grid_pinpoints: list[tuple[int, int]] = [
            (patch_size * i, patch_size * j) for i in range(1, 5) for j in range(1, 5)
        ]

    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain either 'shortest_edge' or 'height' and 'width'."
            )

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _resize_for_patching(
        self,
        image: np.ndarray,
        target_resolution: tuple[int, int],
        resample: PILImageResampling,
        input_data_format: ChannelDimension,
    ) -> np.ndarray:
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image (np.ndarray):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            np.ndarray: The resized and padded image.
        """
        new_height, new_width = _get_patch_output_size(
            image, target_resolution, input_data_format
        )

        # Resize the image
        resized_image = self.resize(
            image,
            size={"height": new_height, "width": new_width},
            resample=resample,
            data_format=input_data_format,
            input_data_format=input_data_format,
        )

        return resized_image

    def _pad_for_patching(
        self,
        image: np.ndarray,
        target_resolution: tuple[int, int],
        input_data_format: ChannelDimension,
    ) -> np.ndarray:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = _get_patch_output_size(
            image, target_resolution, input_data_format
        )

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = pad(image, padding=((paste_y, paste_y), (paste_x, paste_x)))

        if new_width + paste_x * 2 != target_width:
            padded_image = pad(padded_image, padding=((0, 0), (1, 0)))
        if new_height + paste_y * 2 != target_height:
            padded_image = pad(padded_image, padding=((1, 0), (0, 0)))

        return padded_image

    def get_image_patches(
        self,
        image: np.ndarray,
        grid_pinpoints: list[tuple[int, int]],
        size: tuple[int, int],
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> tuple[list[np.ndarray], tuple[int, int]]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (np.ndarray):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            list[np.ndarray]: A list of NumPy arrays containing the processed image patches.
            tuple[int, int]: The number of patches in the image.
        """
        if not isinstance(grid_pinpoints, list):
            raise ValueError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample, input_data_format
        )
        padded_image = self._pad_for_patching(
            resized_image, best_resolution, input_data_format
        )

        patches, num_patches = _divide_to_patches(
            padded_image, patch_size, input_data_format
        )

        # Make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(
                patch, channel_dim=data_format, input_channel_dim=input_data_format
            )
            for patch in patches
        ]

        resized_original_image = resize(
            image,
            size=size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return [resized_original_image] + patches, num_patches

    def preprocess(
        self,
        images: ImageInput,
        size: dict[str, int] = None,
        image_grid_pinpoints: list[tuple[int, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_convert_rgb: bool = None,
        return_tensors: Optional[str] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        if size is not None:
            size = size["shortest_edge"] if "shortest_edge" in size else size["height"]
        else:
            size = self.crop_size
        do_convert_rgb = (
            do_convert_rgb
            if do_convert_rgb is not None
            else self.image_processor.do_convert_rgb
        )
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else self.image_grid_pinpoints
        )

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        new_images: list[np.ndarray] = []
        image_sizes: list[tuple[int, int]] = []
        num_patches: list[tuple[int, int]] = []
        for image in images:
            # Convert an image into a list of patches
            image_patches, num_patches = self.get_image_patches(
                image,
                image_grid_pinpoints,
                size=(self.crop_size["height"], self.crop_size["width"]),
                patch_size=self.crop_size["height"],
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
            )

            # preprocess patches
            preprocess_output = self.image_processor.preprocess(
                image_patches,
                do_resize=False,
                resample=resample,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )
            pixel_values = np.array(preprocess_output["pixel_values"])
            new_images.append(pixel_values)
            image_size = get_image_size(image, channel_dim=input_data_format)
            image_sizes.append(image_size)
            num_patches.append(num_patches)

        # Pad features
        max_patch = max(len(x) for x in new_images)
        pixel_values = [
            np.concatenate(
                [
                    x,
                    np.zeros(
                        [max_patch - x.shape[0]] + list(x.shape[1:]), dtype=x.dtype
                    ),
                ],
                axis=0,
            )
            if x.shape[0] < max_patch
            else x
            for x in new_images
        ]

        data = {
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "num_patches": num_patches,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        return self.image_processor.model_input_names


class MultimodalLanguageModelProcessor(ProcessorMixin):
    """
    MultimodalLanguageModelProcessor is a class that processes text and images for multimodal language models.
    It is a composition of an image processor and a tokenizer.

    Inspired by LLaVA-NeXT, it supports additional techniques for processing
    high resolution images as explained in the [LLaVa paper](https://arxiv.org/abs/2310.03744).
    """

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
        self.image_processor = ImageProcessorWrapper(image_processor)
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        if self.image_processor is not None:
            self.tokenizer.add_tokens("<image>")
            self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        else:
            self.image_token_id = None

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ],
        images: ImageInput = None,
        return_tensors: str | TensorType = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You have to specify either text or images.")

        inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_inputs = self.image_processor(
                images, return_tensors=return_tensors, **kwargs
            )
            inputs.update(image_inputs.data)

        return BatchFeature(data=inputs, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
