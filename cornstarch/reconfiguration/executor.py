"""Executor for dynamic parallel reconfiguration using all-to-all."""

from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn

from cornstarch.shardformer.shard.placeholder import TensorPlaceholder

from .data_structures import LayerOwnership
from .utils import get_all_param_names, get_param_by_name, set_param_by_name


class ReconfigurationExecutor:
    """Executes tensor redistribution using torch.distributed.all_to_all."""

    def __init__(self, model: nn.Module):
        """Initialize the executor.

        Args:
            model: The model to reconfigure
        """
        self.model = model
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def execute(
        self,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership]
    ) -> None:
        """Execute reconfiguration from source to target ownership.

        Args:
            source_ownership: Current ownership mapping (rank -> LayerOwnership)
            target_ownership: Target ownership mapping (rank -> LayerOwnership)
        """
        # Get all parameter names
        all_params = get_all_param_names(source_ownership, target_ownership)

        # Redistribute each parameter
        for param_name in sorted(all_params):  # Sort for deterministic order
            self._redistribute_param(
                param_name,
                source_ownership,
                target_ownership
            )

    def _redistribute_param(
        self,
        param_name: str,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership]
    ) -> None:
        """Redistribute a single parameter using all-to-all.

        To properly use all_to_all, we ensure all tensors at each position
        have the same size by:
        1. Communicating tensor metadata first
        2. Creating appropriately sized tensors (real or dummy)
        3. Using all_to_all for actual transfer
        4. Extracting valid data from received tensors

        Args:
            param_name: Name of the parameter to redistribute
            source_ownership: Source ownership mapping
            target_ownership: Target ownership mapping
        """
        # Step 1: Determine tensor shape and dtype (all ranks need to agree)
        # Get metadata from the rank that has it in source
        tensor_shape = None
        tensor_dtype = None

        if self._is_held_in_source(param_name, self.rank, source_ownership):
            try:
                param = get_param_by_name(self.model, param_name)
                tensor_shape = list(param.shape)
                tensor_dtype = param.dtype
            except KeyError:
                pass

        # Broadcast shape and dtype from rank 0 (or first rank that has it)
        if dist.is_initialized():
            # Gather shapes from all ranks
            shape_list = [None] * self.world_size
            dtype_list = [None] * self.world_size
            dist.all_gather_object(shape_list, tensor_shape)
            dist.all_gather_object(dtype_list, tensor_dtype)

            # Find first non-None shape and dtype
            for s, d in zip(shape_list, dtype_list):
                if s is not None and d is not None:
                    tensor_shape = torch.Size(s)
                    tensor_dtype = d
                    break
        else:
            # Not in distributed mode
            try:
                param = get_param_by_name(self.model, param_name)
                tensor_shape = param.shape
                tensor_dtype = param.dtype
            except KeyError:
                return  # Parameter doesn't exist

        if tensor_shape is None or tensor_dtype is None:
            # Parameter doesn't exist anywhere, skip
            return

        # Step 2: Determine target device from original parameter
        target_device = 'cpu'
        if torch.cuda.is_available():
            try:
                param = get_param_by_name(self.model, param_name)
                target_device = param.device
            except KeyError:
                # Default to current CUDA device
                target_device = f'cuda:{torch.cuda.current_device()}'

        # Step 3: Build input tensor list for all_to_all
        # Each position i contains the tensor to send to rank i
        input_tensor_list = []

        for dst_rank in range(self.world_size):
            if self._should_send_to(param_name, self.rank, dst_rank, source_ownership, target_ownership):
                # Send real tensor to dst_rank
                try:
                    tensor = get_param_by_name(self.model, param_name)
                    input_tensor_list.append(tensor.data.contiguous())
                except KeyError:
                    # Should have tensor but don't (error state), send zeros
                    input_tensor_list.append(torch.zeros(tensor_shape, dtype=tensor_dtype, device=target_device))
            else:
                # Send dummy tensor (zeros) - same size for all_to_all compatibility
                input_tensor_list.append(torch.zeros(tensor_shape, dtype=tensor_dtype, device=target_device))

        # Step 4: Build output tensor list for all_to_all
        # Each position i will receive tensor from rank i
        output_tensor_list = []
        for src_rank in range(self.world_size):
            # Allocate buffer of same size (required for all_to_all)
            output_tensor_list.append(torch.zeros(tensor_shape, dtype=tensor_dtype, device=target_device))

        # Step 5: Execute all-to-all
        if dist.is_initialized():
            dist.all_to_all(output_tensor_list, input_tensor_list)
        else:
            # Not in distributed mode, just copy
            for i, tensor in enumerate(input_tensor_list):
                output_tensor_list[i].copy_(tensor)

        # Step 6: Extract valid received tensors and update model
        # Note: all_to_all_gloo moves tensors through CPU, so ensure they're back on target device
        for src_rank, recv_tensor in enumerate(output_tensor_list):
            if self._should_recv_from(param_name, self.rank, src_rank, source_ownership, target_ownership):
                # This is a valid tensor we need - ensure it's on the correct device
                if recv_tensor.device != target_device:
                    recv_tensor = recv_tensor.to(target_device)
                set_param_by_name(self.model, param_name, recv_tensor)
                break  # Only one rank should send us the tensor

        # Step 6: Replace with placeholder if not needed in target config
        if not self._is_held_in_target(param_name, self.rank, target_ownership):
            self._replace_with_placeholder(param_name)

    def _should_send_to(
        self,
        param_name: str,
        my_rank: int,
        dst_rank: int,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership]
    ) -> bool:
        """Check if this rank should send parameter to dst_rank.

        Args:
            param_name: Parameter name
            my_rank: Current rank
            dst_rank: Destination rank
            source_ownership: Source ownership mapping
            target_ownership: Target ownership mapping

        Returns:
            True if should send
        """
        # I have it in source AND dst needs it in target
        i_have_it = (
            my_rank in source_ownership and
            param_name in source_ownership[my_rank].layer_names and
            not source_ownership[my_rank].is_placeholder.get(param_name, False)
        )
        dst_needs_it = (
            dst_rank in target_ownership and
            param_name in target_ownership[dst_rank].layer_names and
            not target_ownership[dst_rank].is_placeholder.get(param_name, False)
        )
        return i_have_it and dst_needs_it

    def _should_recv_from(
        self,
        param_name: str,
        my_rank: int,
        src_rank: int,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership]
    ) -> bool:
        """Check if this rank should receive parameter from src_rank.

        Args:
            param_name: Parameter name
            my_rank: Current rank
            src_rank: Source rank
            source_ownership: Source ownership mapping
            target_ownership: Target ownership mapping

        Returns:
            True if should receive
        """
        # src has it in source AND I need it in target
        src_has_it = (
            src_rank in source_ownership and
            param_name in source_ownership[src_rank].layer_names and
            not source_ownership[src_rank].is_placeholder.get(param_name, False)
        )
        i_need_it = (
            my_rank in target_ownership and
            param_name in target_ownership[my_rank].layer_names and
            not target_ownership[my_rank].is_placeholder.get(param_name, False)
        )
        return src_has_it and i_need_it

    def _is_held_in_source(
        self,
        param_name: str,
        rank: int,
        source_ownership: Dict[int, LayerOwnership]
    ) -> bool:
        """Check if parameter is held by rank in source config.

        Args:
            param_name: Parameter name
            rank: Rank to check
            source_ownership: Source ownership mapping

        Returns:
            True if parameter is held
        """
        return (
            rank in source_ownership and
            param_name in source_ownership[rank].layer_names and
            not source_ownership[rank].is_placeholder.get(param_name, False)
        )

    def _is_held_in_target(
        self,
        param_name: str,
        rank: int,
        target_ownership: Dict[int, LayerOwnership]
    ) -> bool:
        """Check if parameter should be held by rank in target config.

        Args:
            param_name: Parameter name
            rank: Rank to check
            target_ownership: Target ownership mapping

        Returns:
            True if parameter should be held
        """
        return (
            rank in target_ownership and
            param_name in target_ownership[rank].layer_names and
            not target_ownership[rank].is_placeholder.get(param_name, False)
        )

    def _get_param_metadata(
        self,
        param_name: str,
        rank: int,
        ownership: Dict[int, LayerOwnership]
    ) -> tuple:
        """Get parameter shape and dtype from ownership info.

        Args:
            param_name: Parameter name
            rank: Rank that owns the parameter
            ownership: Ownership mapping

        Returns:
            Tuple of (shape, dtype)
        """
        # Try to get from current model first
        try:
            param = get_param_by_name(self.model, param_name)
            return param.shape, param.dtype
        except KeyError:
            # Parameter doesn't exist locally, use default
            # This is a simplification - in real implementation,
            # we'd need to communicate shape/dtype info
            return torch.Size([1]), torch.float32

    def _replace_with_placeholder(self, param_name: str) -> None:
        """Replace a parameter with a TensorPlaceholder.

        Args:
            param_name: Parameter name
        """
        try:
            param = get_param_by_name(self.model, param_name)
            # Create placeholder
            placeholder = TensorPlaceholder(param)

            # Get parent module
            parts = param_name.split('.')
            module = self.model
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)

            # Store placeholder
            if not hasattr(module, '_parameter_placeholders'):
                module._parameter_placeholders = {}
            module._parameter_placeholders[parts[-1]] = placeholder

            # Set parameter to None
            delattr(module, parts[-1])
            setattr(module, parts[-1], None)

        except KeyError:
            # Parameter doesn't exist, nothing to replace
            pass
