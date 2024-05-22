import numpy as np
from PIL import Image
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from cornstarch.models.multimodal_language_model import MultimodalLanguageModelProcessor


class MultimodalLanguageModelProcessorTest:
    def __init__(self):
        self.batch_size = 7
        self.num_channels = 3
        self.min_resolution = 30
        self.max_resolution = 400
        self.do_resize = True
        self.do_normalize = True
        self.do_convert_rgb = True
        self.size = {"shortest_edge": 20}
        self.image_mean = OPENAI_CLIP_MEAN
        self.image_std = OPENAI_CLIP_STD
