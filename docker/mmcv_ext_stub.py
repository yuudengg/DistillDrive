"""Import-time compatibility shim for unused MMCV 1.x native operators.

DistillDrive uses its own deformable aggregation extension and has no direct
references to ``mmcv.ops``. MMDetection 2.x nevertheless imports every MMCV
operator while its registry is initialized. The original MMCV 1.7.1 native
extension cannot compile against the Blackwell-capable PyTorch 2.13 C++ API.
This module lets registry imports complete, but fails loudly if a runtime path
actually tries to execute one of those unavailable operators.
"""

from __future__ import annotations

from typing import Any, Callable


def __getattr__(name: str) -> Callable[..., Any]:
    def unavailable(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            f"mmcv._ext.{name} is unavailable in the NVIDIA GB10 image. "
            "DistillDrive does not reference mmcv.ops directly; this indicates "
            "that a different MMDetection code path was selected."
        )

    unavailable.__name__ = name
    return unavailable

