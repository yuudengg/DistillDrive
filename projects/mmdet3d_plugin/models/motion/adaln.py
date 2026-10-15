import torch
import torch.nn as nn

from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

NUM_META_ACTIONS = 12 # same as meta_action.NUM_META_ACTIONS (3 lateral x 4 longitudinal)


@PLUGIN_LAYERS.register_module()
class AdaLNZero(BaseModule):
    """adaLN-Zero condition for the feedback refine pass (DiT style).

    meta_idx -> (gamma, beta, alpha), applied as
        out = feature + alpha * ((1 + gamma) * LN(feature) + beta)
    The modulation layer is zero-initialized, so at init alpha = gate_init (0 by default)
    and the output equals the input (feedback off == original refine).
    (1 + gamma) instead of gamma keeps d(out)/d(alpha) = LN(feature) != 0 at init,
    otherwise gamma, beta and alpha all start at 0 and never receive a gradient.
    """

    def __init__(
        self,
        embed_dims=256,
        num_meta=NUM_META_ACTIONS,
        use_uncertainty=False,
        extra_dim=0,
        gate_init=0.0,
    ):
        super(AdaLNZero, self).__init__()
        self.embed_dims = embed_dims
        self.use_uncertainty = use_uncertainty # uncertainty [0, 1] pushes alpha (feedback only when unsure)
        self.extra_dim = extra_dim # > 0: extra condition, e.g. query embedding later
        self.gate_init = gate_init # small positive value if alpha does not grow in training

        self.meta_embed = nn.Embedding(num_meta, embed_dims)
        if extra_dim > 0:
            self.extra_proj = nn.Linear(extra_dim, embed_dims)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dims, embed_dims * 3),
        )
        if use_uncertainty:
            self.uncertainty_proj = nn.Linear(1, embed_dims)
        self.norm = nn.LayerNorm(embed_dims, elementwise_affine=False, eps=1e-6)
        self.last_alpha = None # detached alpha of the last forward, for gate_stats()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.meta_embed.weight, std=0.02)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.constant_(self.modulation[-1].bias[2 * self.embed_dims:], self.gate_init) # alpha
        if self.use_uncertainty:
            nn.init.zeros_(self.uncertainty_proj.weight)
            nn.init.zeros_(self.uncertainty_proj.bias)

    def forward(self, meta_idx, uncertainty=None, extra=None):
        """
        Args:
            meta_idx: [B] (one per sample, from lm.infer_batch), [B, M] (one per planning mode)
                or int (one for the whole batch, e.g. batch 1 at inference), long
            uncertainty: same shape as meta_idx, optional, 0 (sure) ~ 1 (unsure)
            extra: meta_idx.shape + [extra_dim], optional
        Returns:
            gamma, beta, alpha: [B, 1, D] for meta_idx [B], [B, M, D] for meta_idx [B, M],
                [1, 1, D] for int (broadcasts over the batch)
        """
        if not torch.is_tensor(meta_idx) or meta_idx.dim() == 0:
            device = self.meta_embed.weight.device
            meta_idx = torch.as_tensor(meta_idx, device=device).long().reshape(1)
            if uncertainty is not None:
                uncertainty = torch.as_tensor(uncertainty, device=device).float().reshape(1)
            if extra is not None:
                extra = torch.as_tensor(extra, device=device).reshape(1, -1)

        if meta_idx.dim() == 1:
            meta_idx = meta_idx.unsqueeze(1) # [B, 1]
            if uncertainty is not None:
                uncertainty = uncertainty.unsqueeze(1)
            if extra is not None:
                extra = extra.unsqueeze(1)

        cond = self.meta_embed(meta_idx) # [B, M, D]
        if self.extra_dim > 0 and extra is not None:
            cond = cond + self.extra_proj(extra)
        gamma, beta, alpha = self.modulation(cond).chunk(3, dim=-1) # 3 * [B, M, D]
        if self.use_uncertainty and uncertainty is not None:
            alpha = alpha + self.uncertainty_proj(uncertainty.unsqueeze(-1).to(cond.dtype))
        self.last_alpha = alpha.detach()
        return gamma, beta, alpha

    def modulate(self, feature, gamma, beta, alpha):
        """feature [B, N, D], gamma / beta / alpha [B, 1 or N, D] -> [B, N, D]"""
        return feature + alpha * ((1 + gamma) * self.norm(feature) + beta)

    def gate_stats(self):
        """alpha of the last forward, log it during training to check the gate grows"""
        if self.last_alpha is None:
            return {}
        alpha = self.last_alpha.abs()
        return {"alpha_abs_mean": float(alpha.mean()), "alpha_abs_max": float(alpha.max())}
