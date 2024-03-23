import torch


class ParameterPlaceholder(torch.nn.Parameter):
    _shape: torch.Size
    _dtype: torch.dtype
    _device: torch.device

    def __new__(cls, input_tensor: torch.Tensor):
        r: ParameterPlaceholder = super().__new__(
            cls,
            data=torch.empty(0, device="meta"),
            requires_grad=input_tensor.requires_grad,
        )
        r._shape = input_tensor.shape
        r._dtype = input_tensor.dtype
        r._device = input_tensor.device

        return r

    def __repr__(self) -> str:
        return f"ParameterPlaceholder(..., size={tuple(self._shape)}, device={self._device}, dtype={self._dtype})"

    def create(self) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            data=torch.empty(
                self._shape,
                dtype=self._dtype,
                device=self._device,
            ),
            requires_grad=self.requires_grad,
        )

    def to(self, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            dummy_tensor = torch.empty(0, dtype=self._dtype, device=self._device).to(
                *args, **kwargs
            )
            self._device = dummy_tensor.device
            self._dtype = dummy_tensor.dtype

        return self
