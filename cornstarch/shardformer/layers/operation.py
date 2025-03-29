import torch
import torch.distributed as dist

"""
Operations that support non-uniform tensor gathering.
Corresponding Colossal-AI operations assume all tensors have idential shape across ranks,
which is typically not the case in encoder parallelism or arbitrary sequence length.
"""


def _split(
    input_: torch.Tensor,
    sizes: list[int],
    dim: int = -1,
    process_group: dist.ProcessGroup = None,
):
    # skip if only one rank involved
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    tensor_list = torch.split(input_, sizes, dim=dim)
    output = tensor_list[dist.get_rank(process_group)].clone().contiguous()

    return output


def _gather(
    input_: torch.Tensor,
    dim: int = -1,
    process_group: dist.ProcessGroup = None,
    fp8_communication=False,
    fp8_format="e5m2",
):
    assert fp8_communication is False, (
        "FP8 communication is not supported in gather operation. "
        "Please use fp8_communication=False."
    )

    # skip if only one rank involved
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    sizes = [
        torch.empty(1, dtype=torch.int64, device=input_.device)
        for _ in range(world_size)
    ]
    dist.all_gather(
        sizes,
        torch.tensor([input_.size(dim)], device=input_.device),
        group=process_group,
    )
    sizes = [size.item() for size in sizes]

    input_ = input_.contiguous()

    tensor_list = [
        torch.empty(
            (*input_.shape[:dim], size, *input_.shape[(dim + 1) :]),
            dtype=input_.dtype,
            device=input_.device,
        )
        for size in sizes
    ]

    dist.all_gather(tensor_list, input_, group=process_group)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output, sizes


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def forward(
        ctx, input_, dim, process_group, grad_scale=None, fp8_communication=False
    ):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        output, ctx.sizes = _gather(
            input_,
            dim,
            process_group,
            fp8_communication=fp8_communication,
            fp8_format="e4m3",
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale is not None:
            grad_output = grad_output * ctx.grad_scale
        return (
            _split(grad_output, ctx.sizes, ctx.dim, ctx.process_group),
            None,
            None,
            None,
            None,
        )


def gather_forward_split_backward(
    input_, dim, process_group, grad_scale=None, fp8_communication=False
):
    return _GatherForwardSplitBackward.apply(
        input_, dim, process_group, grad_scale, fp8_communication
    )
