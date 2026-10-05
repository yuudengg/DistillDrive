from __future__ import annotations

import importlib
import platform

import torch


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available in the container"

    capability = torch.cuda.get_device_capability()
    print(f"architecture={platform.machine()}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name()} capability={capability}")

    for module_name in ("mmcv", "mmdet", "flash_attn"):
        module = importlib.import_module(module_name)
        print(f"{module_name}={getattr(module, '__version__', 'unknown')}")

    extension = importlib.import_module(
        "projects.mmdet3d_plugin.ops.deformable_aggregation_ext"
    )
    print(f"custom_cuda_extension={extension.__name__}")

    from projects.mmdet3d_plugin.ops import deformable_aggregation_function

    features = torch.arange(
        16, dtype=torch.float32, device="cuda", requires_grad=True
    ).reshape(1, 4, 4)
    spatial_shape = torch.tensor([[[2, 2]]], dtype=torch.int32, device="cuda")
    scale_start_index = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    sampling_location = torch.full(
        (1, 1, 1, 1, 2), 0.5, dtype=torch.float32, device="cuda",
        requires_grad=True,
    )
    weights = torch.ones(
        (1, 1, 1, 1, 1, 1), dtype=torch.float32, device="cuda",
        requires_grad=True,
    )
    output = deformable_aggregation_function(
        features, spatial_shape, scale_start_index, sampling_location, weights
    )
    output.sum().backward()
    torch.cuda.synchronize()
    assert output.shape == (1, 1, 4)
    assert torch.isfinite(output).all()
    assert sampling_location.grad is not None
    assert weights.grad is not None
    print(f"custom_cuda_forward_backward=ok output={output.detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
