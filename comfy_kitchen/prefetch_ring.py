from __future__ import annotations

from collections.abc import Callable

import torch

_record_region: Callable[[torch.Tensor], None] | None = None


def set_record_region(callback: Callable[[torch.Tensor], None] | None) -> None:
    global _record_region
    _record_region = callback


def recording() -> bool:
    return _record_region is not None


def record_region(tensor: torch.Tensor) -> None:
    """Append `tensor`'s bytes to the step's read-order list (no-op unless recording)."""
    if _record_region is not None:
        _record_region(tensor)


def _cuda():
    # imported lazily: the CUDA backend records read order through this module
    from .backends import cuda as _cuda_backend

    return _cuda_backend


def is_available() -> bool:
    cuda = _cuda()
    ext = cuda._C if cuda._EXT_AVAILABLE else None
    return bool(ext is not None and hasattr(ext, "prefetch_ring_available") and ext.prefetch_ring_available())


def configure(regions: torch.Tensor, count: int, lookahead_bytes: int, chunk_bytes: int = 96 * 1024) -> None:
    cuda = _cuda()
    stream = torch.cuda.current_stream(regions.device).cuda_stream
    cuda._C.prefetch_ring_configure(cuda._wrap_for_dlpack(regions), count, lookahead_bytes, chunk_bytes, stream)


def disable(device: torch.device | int | None = None) -> None:
    cuda = _cuda()
    if not cuda._EXT_AVAILABLE or not hasattr(cuda._C, "prefetch_ring_disable"):
        return
    stream = torch.cuda.current_stream(device).cuda_stream
    cuda._C.prefetch_ring_disable(stream)


def start(device: torch.device | int | None = None) -> None:
    """Launch the issuer for one decode step; it runs alongside the step's kernels on a side stream."""
    stream = torch.cuda.current_stream(device).cuda_stream
    _cuda()._C.prefetch_ring_start(stream)


def stats() -> tuple[int, int, int, int, int, int, list[int], list[int]]:
    """(total, consumed, stalled, touched, skipped, waited_ns, smids, distinct_hist) of the current
    device's ring; touched/skipped/waited_ns are cumulative over steps, smids is the SM of each
    issuer CTA in the last step, distinct_hist[n] the number of steps whose issuer CTAs landed on
    n distinct SMs. Synchronizes the device."""
    torch.cuda.synchronize()
    return _cuda()._C.prefetch_ring_stats()
