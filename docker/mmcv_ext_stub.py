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

# Purely informational calls (env logging via mmdet.utils.collect_env, etc.)
# that mmdet's own startup code invokes regardless of whether any actual
# mmcv.ops kernel is used. Safe to answer with a placeholder instead of
# raising, since no computation depends on the result.
_INFO_ONLY = {
    "get_compiler_version": lambda: "n/a (mmcv native ops unavailable on GB10)",
    "get_compiling_cuda_version": lambda: "n/a (mmcv native ops unavailable on GB10)",
}


def __getattr__(name: str) -> Callable[..., Any]:
    if name in _INFO_ONLY:
        return _INFO_ONLY[name]

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            f"mmcv._ext.{name} is unavailable in the NVIDIA GB10 image. "
            "DistillDrive does not reference mmcv.ops directly; this indicates "
            "that a different MMDetection code path was selected."
        )

    unavailable.__name__ = name
    return unavailable

