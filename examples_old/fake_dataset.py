import torch
from PIL import Image
from torch.utils.data import Dataset


class FakeDataset(Dataset):
    def __init__(self, image_size: tuple[int, int]):
        random_image = torch.randint(0, 256, image_size + (3,), dtype=torch.uint8)
        self.image = Image.fromarray(random_image.numpy())
        self.text = "<image>" + " text" * 256

    def __len__(self):
        return 65536

    def __getitem__(self, index: int) -> dict:
        return {"image": self.image, "text": self.text}
