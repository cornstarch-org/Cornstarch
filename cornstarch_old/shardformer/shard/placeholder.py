import torch
from colossalai.accelerator import get_accelerator


class TensorPlaceholder:
    param_id: int
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool

    def __init__(self, input_tensor: torch.Tensor):
        self.param_id = id(input_tensor)
        self.shape = input_tensor.shape
        self.dtype = input_tensor.dtype
        self.requires_grad = input_tensor.requires_grad

    def __repr__(self) -> str:
        return f"TensorPlaceholder(..., id={self.param_id})"

    def create(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        device = get_accelerator().get_current_device() if device is None else device
        dtype = self.dtype if dtype is None else dtype
        return torch.empty(
            self.shape, device=device, dtype=dtype, requires_grad=self.requires_grad
        )
