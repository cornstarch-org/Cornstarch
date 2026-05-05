from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=None)
def get_hf_kernel(kernel_id: str, version: int | None = None) -> Any:
    try:
        from kernels import get_kernel
    except ImportError as exc:
        raise ImportError(
            "The `kernels` package is required to load Hugging Face kernels. "
            "Install it or use a test stub before calling load_kernel()."
        ) from exc

    if version is None:
        return get_kernel(kernel_id)
    return get_kernel(kernel_id, version=version)
