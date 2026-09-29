"""Appended to mmcv/ops/focal_loss.py at image build time.

mmcv-full 1.7.1's sigmoid_focal_loss/softmax_focal_loss are backed by a
compiled CUDA extension that does not exist in this image (mmcv's native
ops were not buildable against this PyTorch, see mmcv_ext_stub.py).
DistillDrive itself never calls mmcv.ops directly, but mmdet's built-in
FocalLoss does, whenever a config uses `type='FocalLoss'` on a CUDA tensor.

This redefines the two module-level names mmdet imports
(`from mmcv.ops import sigmoid_focal_loss`) with plain-PyTorch, fully
autograd-differentiable equivalents that match the compiled kernels'
exact calling convention (target as a class-index LongTensor with
num_classes meaning "background", not one-hot; optional per-class
weight; reduction 'none'/'mean'/'sum' with 'mean' dividing by N).
Since this file is executed top-to-bottom at import time, these
reassignments simply override the `Function.apply` versions defined
above, with no other code path changes required.
"""

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(input, target, gamma=2.0, alpha=0.25, weight=None,
                        reduction='mean'):
    assert target.dtype == torch.long
    assert input.dim() == 2
    assert target.dim() == 1
    num_classes = input.size(1)
    target_onehot = F.one_hot(
        target, num_classes=num_classes + 1)[:, :num_classes].type_as(input)
    pred_sigmoid = input.sigmoid()
    pt = (1 - pred_sigmoid) * target_onehot + pred_sigmoid * (1 - target_onehot)
    focal_weight = (alpha * target_onehot + (1 - alpha) *
                     (1 - target_onehot)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        input, target_onehot, reduction='none') * focal_weight
    if weight is not None and weight.numel() > 0:
        loss = loss * weight
    if reduction == 'mean':
        loss = loss.sum() / input.size(0)
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def softmax_focal_loss(input, target, gamma=2.0, alpha=0.25, weight=None,
                        reduction='mean'):
    assert target.dtype == torch.long
    assert input.dim() == 2
    assert target.dim() == 1
    pred_softmax = input.softmax(dim=1)
    pt = pred_softmax.gather(1, target.unsqueeze(1)).squeeze(1)
    target_weight = alpha * torch.ones_like(pt)
    focal_weight = target_weight * (1 - pt).pow(gamma)
    loss = F.cross_entropy(input, target, reduction='none') * focal_weight
    if weight is not None and weight.numel() > 0:
        loss = loss * weight[target]
    if reduction == 'mean':
        loss = loss.sum() / input.size(0)
    elif reduction == 'sum':
        loss = loss.sum()
    return loss
