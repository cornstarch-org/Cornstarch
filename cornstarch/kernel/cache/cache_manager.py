from typing import Dict, Optional, Tuple

import torch
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.compile_tasks import _module_to_triton_kernel
from torch.nn.attention.flex_attention import BlockMask


class FlexAttnKernelConfig:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        sm_scale: float,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.sm_scale = sm_scale

    def __hash__(self):
        return hash(
            (
                self.batch_size,
                self.num_heads,
                self.seq_len,
                self.head_dim,
                self.dtype,
                self.sm_scale,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, FlexAttnKernelConfig):
            return False

        return (
            self.batch_size == other.batch_size
            and self.num_heads == other.num_heads
            and self.seq_len == other.seq_len
            and self.head_dim == other.head_dim
            and self.dtype == other.dtype
            and self.sm_scale == other.sm_scale
        )


class CacheManager:
    """
    Manage the cache directory for torch inductor and FlexAttention kernels.
    """

    def __init__(self):
        self.kernel_cache: Dict[FlexAttnKernelConfig, Tuple[object, object]] = {}
        self.cached_keys = set()  # Store cache keys

    def clear_cache(self):
        self.kernel_cache.clear()
        self.cached_keys.clear()

    def get_cached_kernels(
        self, config: FlexAttnKernelConfig
    ) -> Optional[Tuple[object, object]]:
        """Get cached forward and backward kernels if they exist."""
        return self.kernel_cache.get(config, None)

    def cache_kernels(
        self,
        config: FlexAttnKernelConfig,
        fwd_kernel: object,
        bwd_kernel: object,
        fwd_key: str,
        bwd_key: str,
    ):
        """Cache the forward and backward kernels for a given configuration."""
        if fwd_key not in self.cached_keys and bwd_key not in self.cached_keys:
            self.kernel_cache[config] = (fwd_kernel, bwd_kernel)
            self.cached_keys.add(fwd_key)
            self.cached_keys.add(bwd_key)
        else:
            raise ValueError(f"Kernel for {config} already cached")

    def extract_kernel_from_inductor_cache(self):
        """Extract FlexAttention kernels from PyCodeCache."""
        fwd_kernel = None
        bwd_kernel = None
        fwd_key = None
        bwd_key = None

        for key, mod in PyCodeCache.cache.items():
            if key in self.cached_keys:
                continue
            try:
                fwd_kernel = _module_to_triton_kernel(mod, "triton_flex_attention")
                fwd_key = key
            except Exception:
                pass
            try:
                bwd_kernel = _module_to_triton_kernel(
                    mod, "triton_flex_attention_backward"
                )
                bwd_key = key
            except Exception:
                pass

            if fwd_kernel and bwd_kernel:
                break

        return fwd_kernel, bwd_kernel, fwd_key, bwd_key

    def warmup(
        self,
        config: FlexAttnKernelConfig,
        block_mask: Optional[BlockMask] = None,
    ):
        """Warmup run to cache the kernels."""
        batch_size, num_heads, seq_len, head_dim, dtype, sm_scale = (
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim,
            config.dtype,
            config.sm_scale,
        )

        # Create dummy inputs with requires_grad=True
        q = torch.empty(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=dtype,
            device="cuda",
        ).requires_grad_()
        k = torch.empty(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=dtype,
            device="cuda",
        ).requires_grad_()
        v = torch.empty(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=dtype,
            device="cuda",
        ).requires_grad_()

        # Check if already cached
        if self.get_cached_kernels(config):
            return

        # Run kernel to cache
        from torch.nn.attention.flex_attention import flex_attention

        torch.set_grad_enabled(True)
        flexattention = torch.compile(
            flex_attention, backend="inductor", fullgraph=True
        )

        # Forward pass
        flex_out, softmax_lse = flexattention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=sm_scale,
            return_lse=True,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )

        # Create gradient for backward pass
        dout = torch.randn_like(flex_out).requires_grad_()
        flex_out.backward(dout, retain_graph=True)

        # Extract and cache kernels
        fwd_kernel, bwd_kernel, fwd_key, bwd_key = (
            self.extract_kernel_from_inductor_cache()
        )
        # print(fwd_kernel, bwd_kernel, fwd_key, bwd_key)
        if fwd_kernel and bwd_kernel:
            # fwd_kernel = flexattention
            self.cache_kernels(config, fwd_kernel, bwd_kernel, fwd_key, bwd_key)
        else:
            raise ValueError("Failed to extract kernels from inductor cache")


# Private Global cache manager instance
CACHE_MANAGER = CacheManager()


def get_cached_kernels(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    sm_scale: float = 0.5,
    block_mask: Optional[BlockMask] = None,
) -> Optional[Tuple[object, object]]:
    config = FlexAttnKernelConfig(
        batch_size, num_heads, seq_len, head_dim, dtype, sm_scale
    )
    if not CACHE_MANAGER.get_cached_kernels(config):
        CACHE_MANAGER.warmup(config, block_mask)
    return CACHE_MANAGER.get_cached_kernels(config)
