import pytest
import torch
from colossalai.accelerator import get_accelerator
from cornstarch.shardformer.shard.placeholder import TensorPlaceholder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "target_device, target_dtype",
    [
        [torch.device("cuda:0"), torch.float32],
        [torch.device("cuda:0"), torch.float16],
        [torch.device("cpu"), torch.float32],
    ],
    ids=["cuda_fp32", "cuda_fp16", "cpu_fp32"],
)
@pytest.mark.parametrize(
    "size",
    [torch.Size([1, 1]), torch.Size([512, 1, 1]), torch.Size([1024, 2, 1024])],
    ids=["1x1", "512x1x1", "1024x2x1024"],
)
def test_implement_parameter_placeholder(
    target_device: torch.device, target_dtype: torch.dtype, size: torch.Size
):
    input_tensor = torch.empty(size, dtype=torch.float32, device=torch.device("cpu"))
    placeholder: TensorPlaceholder = TensorPlaceholder(input_tensor)

    assert placeholder.shape == size
    assert placeholder.dtype == torch.float32
    assert placeholder.param_id == id(input_tensor)

    new_tensor = placeholder.create(dtype=target_dtype, device=target_device)
    assert isinstance(new_tensor, torch.Tensor)
    assert new_tensor.shape == size
    assert new_tensor.dtype == target_dtype
    assert new_tensor.device == target_device

    new_tensor = placeholder.create()
    assert isinstance(new_tensor, torch.Tensor)
    assert new_tensor.shape == size
    assert new_tensor.dtype == input_tensor.dtype
    assert new_tensor.device == get_accelerator().get_current_device()
