from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RuntimeInfo:
    backend: str
    device: str
    cuda_available: bool
    estimated_vram_gb: float | None


def inspect_runtime() -> RuntimeInfo:
    try:
        import torch  # type: ignore
    except Exception:
        return RuntimeInfo(
            backend="numpy",
            device="cpu",
            cuda_available=False,
            estimated_vram_gb=None,
        )

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return RuntimeInfo(
            backend="torch",
            device="cpu",
            cuda_available=False,
            estimated_vram_gb=None,
        )

    props: Any = torch.cuda.get_device_properties(0)
    total_memory = getattr(props, "total_memory", 0)
    return RuntimeInfo(
        backend="torch",
        device="cuda:0",
        cuda_available=True,
        estimated_vram_gb=float(total_memory) / (1024.0**3),
    )


def fits_vram_budget(max_vram_gb: float) -> bool:
    info = inspect_runtime()
    if info.estimated_vram_gb is None:
        return True
    return info.estimated_vram_gb <= max_vram_gb + 1e-6
