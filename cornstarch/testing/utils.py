import io

import numpy as np
from PIL import Image


def create_random_image(width: int, height: int) -> Image:
    """
    Creates a fake JPEG image with random noise of specified width and height.

    Parameters:
    width (int): The width of the image.
    height (int): The height of the image.

    Returns:
    PIL.JpegImagePlugin.JpegImageFile: An instance of a JPEG image with random noise.
    """
    # Generate random noise for the image (values between 0 and 255 for RGB channels)
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Create an image from the random data
    image = Image.fromarray(random_data, "RGB")

    # Save the image to a BytesIO object to simulate a JPEG file
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Open the image from the BytesIO object as a JpegImageFile instance
    jpeg_image = Image.open(image_bytes)

    return jpeg_image
