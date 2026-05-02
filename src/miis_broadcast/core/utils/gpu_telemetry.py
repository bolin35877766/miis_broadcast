"""
CUDA VRAM telemetry for stdout / CLIENT_DIAG lines (uses PyTorch; no extra pip deps).

Uses ``torch.cuda.mem_get_info`` (driver-wide used = total - free on the device)
and ``torch.cuda.memory_allocated`` (this process, PyTorch caching allocator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CudaVramSnapshot:
    """VRAM stats for one CUDA device index."""

    used_mib: float
    total_mib: float
    used_pct: float
    torch_alloc_mib: float


def cuda_vram_snapshot(device: int = 0) -> Optional[CudaVramSnapshot]:
    """
    Return VRAM snapshot for ``device``, or None if CUDA is unavailable or query fails.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
    except Exception:
        return None
    if total_b <= 0:
        return None
    used_b = total_b - free_b
    used_pct = 100.0 * float(used_b) / float(total_b)
    try:
        torch_alloc_b = torch.cuda.memory_allocated(device)
    except Exception:
        torch_alloc_b = 0
    return CudaVramSnapshot(
        used_mib=used_b / (1024.0**2),
        total_mib=total_b / (1024.0**2),
        used_pct=used_pct,
        torch_alloc_mib=float(torch_alloc_b) / (1024.0**2),
    )


def vram_log_suffix(snap: Optional[CudaVramSnapshot]) -> str:
    """
    Uniform trailing segment for [Client] / [Server] telemetry lines.
    Global VRAM usage (driver) + PyTorch bytes allocated in this process.
    """
    if snap is None:
        return " | GPU_VRAM=n/a"
    return (
        f" | GPU_VRAM={snap.used_mib:.0f}/{snap.total_mib:.0f} MiB ({snap.used_pct:.0f}%) "
        f"torch_alloc={snap.torch_alloc_mib:.0f} MiB"
    )


def vram_log_suffix_from_wire(
    used_mib: Optional[float],
    total_mib: Optional[float],
    torch_alloc_mib: Optional[float],
) -> str:
    """Rebuild log suffix from CLIENT_DIAG JSON fields (omitted or null => n/a)."""
    if used_mib is None or total_mib is None or total_mib <= 0:
        return " | GPU_VRAM=n/a"
    pct = 100.0 * float(used_mib) / float(total_mib)
    ta = 0.0 if torch_alloc_mib is None else float(torch_alloc_mib)
    return (
        f" | GPU_VRAM={float(used_mib):.0f}/{float(total_mib):.0f} MiB ({pct:.0f}%) "
        f"torch_alloc={ta:.0f} MiB"
    )
