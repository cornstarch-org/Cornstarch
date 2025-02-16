from typing import Optional

import torch
import torch.distributed as dist


def batch_isend_irecv_gloo(p2p_op_list: list[dist.P2POp]) -> list[dist.Work]:
    reqs: list[tuple[dist.Work, torch.Tensor]] = []
    for p2p_op in p2p_op_list:
        if p2p_op.op == dist.isend:
            tensor = p2p_op.tensor.to("cpu")
            work = p2p_op.op(tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        else:
            tensor = torch.empty_like(p2p_op.tensor, device="cpu")
            work = p2p_op.op(tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)

        reqs.append((work, tensor))

    send_reqs = []
    with torch.no_grad():
        for (req, tensor), p2p_op in zip(reqs, p2p_op_list):
            if req is None:
                continue

            if p2p_op.op == dist.irecv:
                req.wait()
                p2p_op.tensor.copy_(tensor)
            else:
                send_reqs.append(req)

    return send_reqs


def all_to_all_gloo(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
    async_op: Optional[bool] = False,
):
    """Backend gloo doesn't support all_to_all, so we simulate it here."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    # Each rank gathers the "i-th" input from all ranks
    for i in range(world_size):
        chunk = input_tensor_list[i].to("cpu")  # Each rank's i-th input tensor
        gathered_tensors = [
            torch.empty_like(chunk) for _ in range(world_size)
        ]  # Buffers for allgather

        # Perform all_gather to collect the i-th tensor from all ranks
        dist.all_gather(gathered_tensors, chunk, group)

        if i == rank:
            assert len(output_tensor_list) == len(gathered_tensors)
            for output_tensor, gathered_tensor in zip(
                output_tensor_list, gathered_tensors
            ):
                output_tensor.copy_(gathered_tensor)


def all_to_all_single_gloo(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: Optional[bool] = False,
):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    chunk_size = input.size(0) // world_size
    gathered_tensors = [torch.empty_like(input) for _ in range(world_size)]

    dist.all_gather(gathered_tensors, input, group)

    output_tensor = torch.cat(
        [
            gathered_tensors[i][rank * chunk_size : (rank + 1) * chunk_size]
            for i in range(world_size)
        ],
        dim=0,
    )
    output.copy_(output_tensor)


def reduce_scatter_gloo(
    output: torch.Tensor,
    input_list: list[torch.Tensor],
    op=dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    """
    Implements reduce_scatter using supported PyTorch distributed APIs.
    Args:
        output (torch.Tensor): The tensor to store the scattered reduced result.
        input_list (list[torch.Tensor]): List of input tensors from each process.
        op (dist.ReduceOp, optional): The reduction operation (e.g., SUM, PROD). Defaults to dist.ReduceOp.SUM.
        group (dist.ProcessGroup, optional): The process group to work on. Defaults to None.
        async_op (bool, optional): If set to True, performs the operation asynchronously. Defaults to False.
    Returns:
        dist.Work or None: If async_op is True, returns a Work object. Otherwise, returns None.
    """
    assert op in [
        dist.ReduceOp.SUM,
        dist.ReduceOp.AVG,
    ], f"Unsupported reduce operation: {op.name}"

    world_size = dist.get_world_size(group=group)
    my_rank = dist.get_rank(group=group)
    # Ensure that input_list has tensors from all processes
    if len(input_list) != world_size:
        raise ValueError(f"input_list must contain {world_size} tensors.")

    global_tensor = torch.cat(input_list, dim=0)
    dist.all_reduce(global_tensor, op=op, group=group)
    chunks = torch.chunk(global_tensor, world_size, dim=0)

    output.copy_(chunks[my_rank])
