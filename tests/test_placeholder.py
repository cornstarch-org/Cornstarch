import pytest
import torch

from oobleck_colossalai.shardformer.shard.placeholder import ParameterPlaceholder


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
    placeholder: ParameterPlaceholder = ParameterPlaceholder(input_tensor)

    assert placeholder._shape == size
    assert placeholder._dtype == torch.float32
    assert placeholder._device == torch.device("cpu")
    assert placeholder.device == torch.device("meta")

    placeholder = placeholder.to(dtype=target_dtype, device=target_device)
    assert placeholder._shape == size
    assert placeholder._dtype == target_dtype
    assert placeholder._device == target_device
    assert placeholder.device == torch.device("meta")

    output_parameter = placeholder.create()
    assert isinstance(output_parameter, torch.nn.Parameter)
    assert output_parameter.shape == size
    assert output_parameter.dtype == target_dtype
    assert output_parameter.device == target_device
