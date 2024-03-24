import torch


class ParameterPlaceholder(torch.nn.Parameter):
    _shape: torch.Size
    _dtype: torch.dtype
    _device: torch.device
    old_param_id: int

    def __new__(cls, input_param: torch.nn.Parameter):
        r: ParameterPlaceholder = super().__new__(
            cls,
            data=torch.empty(0, device="meta"),
            requires_grad=input_param.requires_grad,
        )
        r._shape = input_param.shape
        r._dtype = input_param.dtype
        r._device = input_param.device
        r.old_param_id = id(input_param)

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
