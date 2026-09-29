"""Appended to mmcv/parallel/distributed.py at image build time.

mmcv-full 1.7.1's MMDistributedDataParallel._run_ddp_forward references
self._use_replicated_tensor_module / self._replicated_tensor_module, an
experimental PyTorch DDP feature that existed for a few releases around
1.12-2.0 and has since been removed entirely from torch.nn.parallel's
DistributedDataParallel. On this image's torch 2.13 that attribute lookup
raises AttributeError the first time evaluation runs (validation is the
only code path that calls model(...) directly instead of train_step()).

This reassigns the method to always use self.module, which is what the
original code did whenever that experimental feature was disabled (the
common case, and the only one this single-GPU setup exercises).
"""

from typing import Any


def _run_ddp_forward(self, *inputs, **kwargs) -> Any:
    module_to_run = self.module
    if self.device_ids:
        inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
        return module_to_run(*inputs[0], **kwargs[0])
    else:
        return module_to_run(*inputs, **kwargs)


MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward
