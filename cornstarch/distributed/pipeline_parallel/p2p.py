"""Generic pipeline P2P communication.

Objects are serialized with PyTorch's ``c10d._object_to_tensor`` /
``c10d._tensor_to_object`` and exchanged in two phases: a size-header
exchange (so the receiver knows how many bytes to allocate) followed by
the actual data exchange.  Both phases use ``dist.batch_isend_irecv``
so sender and receiver issue their ops simultaneously, avoiding deadlock.
The implementation is backend-agnostic (gloo in tests, nccl in production).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh


def _serialize(obj: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Serialize ``obj`` to a ``(data_tensor, size_tensor)`` pair on ``device``."""
    data, size = c10d._object_to_tensor(obj, device=device, group=dist.GroupMember.WORLD)
    return data.to(device), size.to(device)


def _deserialize(data: torch.Tensor, size: int) -> Any:
    """Deserialize a byte tensor back to a Python object."""
    return c10d._tensor_to_object(data.cpu(), size, group=dist.GroupMember.WORLD)


def _build_p2p_ops(
    send_tensor: torch.Tensor | None,
    send_ranks: list[int],
    recv_tensors: list[torch.Tensor],
    recv_ranks: list[int],
    send_first: bool,
) -> list[dist.P2POp]:
    """Build an ordered list of P2POps respecting ``send_first`` ordering."""
    send_ops = []
    if send_tensor is not None:
        send_ops = [dist.P2POp(dist.isend, send_tensor, r) for r in send_ranks]
    recv_ops = [
        dist.P2POp(dist.irecv, recv_tensors[i], r) for i, r in enumerate(recv_ranks)
    ]
    if send_first:
        return send_ops + recv_ops
    return recv_ops + send_ops


def exchange_objects(
    send_obj: Any | None,
    send_ranks: list[int],
    recv_ranks: list[int],
    device: torch.device,
    send_first: bool = True,
) -> list[Any]:
    """Send ``send_obj`` to ``send_ranks`` and receive one object per ``recv_ranks``.

    A backend-agnostic, mesh-free counterpart to
    :meth:`PipelineP2PCommunication._exchange`: the peer ranks are passed
    explicitly as **global** ranks rather than derived from a mesh, so this is
    the transport used for cross-mesh execution-plan edges (a node on one
    modality's mesh feeding a node on another's).  The same two-phase protocol
    (size header, then payload) and the same ``send_first`` deadlock-avoidance
    ordering are used as the pipeline-stage path.  Returns the received objects in
    ``recv_ranks`` order (empty when ``recv_ranks`` is empty).
    """
    send_data: torch.Tensor | None = None
    send_size: torch.Tensor | None = None
    if send_obj is not None and send_ranks:
        send_data, send_size = _serialize(send_obj, device)

    # Phase 1 — exchange sizes.
    recv_sizes = [
        torch.zeros(1, dtype=torch.long, device=device) for _ in recv_ranks
    ]
    ops = _build_p2p_ops(send_size, send_ranks, recv_sizes, recv_ranks, send_first)
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    # Phase 2 — exchange data.
    recv_bufs = [
        torch.empty(recv_sizes[i].item(), dtype=torch.uint8, device=device)
        for i in range(len(recv_ranks))
    ]
    ops = _build_p2p_ops(send_data, send_ranks, recv_bufs, recv_ranks, send_first)
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    return [
        _deserialize(recv_bufs[i], recv_sizes[i].item())
        for i in range(len(recv_ranks))
    ]


class PipelineP2PCommunication:
    """Backend-agnostic PP point-to-point communication.

    Ranks are resolved from the ``ModalProcessGroupMesh`` rather than
    hard-coded, so the same class handles both intra-modality hops and
    cross-modality boundaries without specialization.
    """

    def __init__(self, mesh: ModalProcessGroupMesh) -> None:
        self._mesh = mesh
        self._device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else "cpu"
        )

    def _exchange(
        self,
        send_obj: Any | None,
        send_ranks: list[int],
        recv_ranks: list[int],
        send_first: bool = True,
    ) -> list[Any]:
        """Send ``send_obj`` to ``send_ranks`` and receive from ``recv_ranks``."""
        return exchange_objects(
            send_obj, send_ranks, recv_ranks, self._device, send_first
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recv_forward(self) -> Any | None:
        """Receive forward activations from the previous stage."""
        prev_ranks = self._mesh.get_prev_ranks()
        if not prev_ranks:
            return None
        received = self._exchange(None, [], prev_ranks)
        return received[0] if len(received) == 1 else received

    def send_forward(self, output: Any) -> None:
        """Send forward activations to the next stage."""
        next_ranks = self._mesh.get_next_ranks()
        if not next_ranks:
            return
        self._exchange(output, next_ranks, [])

    def recv_backward(self) -> Any | None:
        """Receive backward gradients from the next stage."""
        next_ranks = self._mesh.get_next_ranks()
        if not next_ranks:
            return None
        received = self._exchange(None, [], next_ranks)
        return received[0] if len(received) == 1 else received

    def send_backward(self, grad: Any) -> None:
        """Send backward gradients to the previous stage."""
        prev_ranks = self._mesh.get_prev_ranks()
        if not prev_ranks:
            return
        self._exchange(grad, prev_ranks, [])

    def send_forward_recv_backward(
        self, output: Any, send_first: bool = True
    ) -> Any | None:
        """Send forward activations, then receive backward gradients.

        The object protocol has separate size and payload phases. Completing the
        forward object before starting the backward object prevents those phases
        from crossing when an H2 neighbor is still in its deeper warmup.
        """
        self.send_forward(output)
        return self.recv_backward()

    def send_backward_recv_forward(
        self, grad: Any, send_first: bool = False
    ) -> Any | None:
        """Receive forward activations while sending backward gradients.

        Receive-first ordering is required by the deeper H2 warmup: the previous
        stage may still be in a forward-only warmup send and cannot receive this
        gradient until that activation transfer completes.
        """
        forward = self.recv_forward()
        self.send_backward(grad)
        return forward
