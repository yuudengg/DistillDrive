"""Sparse query pruning helpers for the DistillDrive planning decoder.

Paper terms (ICCV 2025, Fig. 2 / Sec. 3.3):
    N_A : number of agent instances coming out of the Perception Model (900)
    N_E : number of ego planning modes (3 commands x 6 modes = 18)
    The Temporal / Agent / Map Decoder and the Generative Decoder all run on the
    same query tensor of length N_A + N_E. These helpers shrink N_A -> K for that
    tensor only. The Perception Model and the Memory Bank keep all N_A slots.

Everything here is pure PyTorch (no mmcv / mmdet) so it can be unit-tested alone.
"""
from typing import List, Optional, Sequence, Tuple

import torch

Tensor = torch.Tensor

# ─────────────────────────────────────────────────────────────────────────────
# gather / scatter along the token axis
# ─────────────────────────────────────────────────────────────────────────────


def _expand_index(idx: Tensor, like: Tensor) -> Tensor:
    """idx [B, K] -> [B, K, *like.shape[2:]] for gather/scatter on dim 1."""
    B, K = idx.shape
    view = idx.view(B, K, *([1] * (like.dim() - 2)))
    return view.expand(B, K, *like.shape[2:])


def gather_tokens(x: Tensor, idx: Tensor) -> Tensor:
    """x [B, N, *S], idx [B, K] -> [B, K, *S]."""
    return x.gather(1, _expand_index(idx, x))


def scatter_tokens(src: Tensor, idx: Tensor, num_full: int, fill: float) -> Tensor:
    """Inverse of gather_tokens. src [B, K, *S] -> [B, num_full, *S].

    Slots not in idx are filled with `fill`. Non in-place, so autograd flows to src.
    """
    out = src.new_full((src.shape[0], num_full, *src.shape[2:]), fill)
    return out.scatter(1, _expand_index(idx, src), src)


def append_ego_slots(keep_idx: Tensor, num_agent_full: int, num_ego: int) -> Tensor:
    """[B, K] agent slots -> [B, K + N_E] slots in the full (N_A + N_E) layout."""
    B = keep_idx.shape[0]
    ego = torch.arange(num_agent_full, num_agent_full + num_ego, device=keep_idx.device)
    return torch.cat([keep_idx, ego.unsqueeze(0).expand(B, -1)], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# query selection
# ─────────────────────────────────────────────────────────────────────────────


def gt_proximity_mask(anchor_xy: Tensor, gt_xy: Sequence[Tensor], radius: float) -> Tensor:
    """Mark, for every GT box, the nearest predicted anchor in BEV (if within radius).

    Used at train time as a stand-in for the Hungarian match, which the det head
    only computes later inside loss() (det_head.sampler.indices is stale in forward).

    anchor_xy: [B, N, 2]   gt_xy: list of [n_b, >=2]   ->   [B, N] bool
    """
    B, N, _ = anchor_xy.shape
    out = torch.zeros(B, N, dtype=torch.bool, device=anchor_xy.device)
    for b in range(B):
        g = gt_xy[b]
        if g is None or len(g) == 0:
            continue
        g = g[:, :2].to(device=anchor_xy.device, dtype=anchor_xy.dtype)
        dist = torch.cdist(g.unsqueeze(0), anchor_xy[b : b + 1].detach()).squeeze(0)  # [n, N]
        dmin, amin = dist.min(dim=1)
        out[b, amin[dmin < radius]] = True
    return out


def gru_source_mask(row_mask_full: Tensor, num_agent_full: int, num_layers: int) -> Tensor:
    """Agent slots whose features feed the GRU h0 of the given rows, under the
    ORIGINAL Generative Decoder reshape (motion_planning_head.py, future_states_predict):

        hidden_state = hidden_states.clone().reshape(L, -1, D // L)   # [L, B*N1, D/L]

    That reshape does not split each token into L chunks; it slices the flattened
    [B*N1*D] buffer, so row r at GRU layer l reads token (l * B * N1 + r) // L.
    With B=1, N1=918, the ego rows (900..917) read agent slots 225-229, 454-458,
    684-688. With B>1 the sources cross into other samples of the batch.

    row_mask_full: [B, N1_full] bool (rows whose output must be reproduced exactly)
    returns:       [B, num_agent_full] bool (agent slots that must therefore be kept)
    """
    B, N1 = row_mask_full.shape
    total = B * N1
    rows = row_mask_full.reshape(-1).nonzero(as_tuple=True)[0]
    out = torch.zeros(B, num_agent_full, dtype=torch.bool, device=row_mask_full.device)
    if rows.numel() == 0:
        return out
    layers = torch.arange(num_layers, device=rows.device)
    tok = ((layers[:, None] * total + rows[None, :]) // num_layers).reshape(-1)
    b, s = tok // N1, tok % N1
    is_agent = s < num_agent_full
    out[b[is_agent], s[is_agent]] = True
    return out


def select_queries(
    det_confidence: Tensor,
    num_query: int,
    must_keep: Optional[Tensor] = None,
) -> Tuple[Tensor, int]:
    """Top-K agent slots by detection confidence, with a must-keep override.

    K_eff = max(num_query, max_b |must_keep_b|), so every must-keep slot survives.
    Returned indices are sorted ascending to preserve the original slot order.

    det_confidence: [B, N]   must_keep: [B, N] bool   ->   keep_idx [B, K_eff], K_eff
    """
    B, N = det_confidence.shape
    score = det_confidence.detach().float()
    k = num_query
    if must_keep is not None and must_keep.any():
        k = max(k, int(must_keep.sum(dim=1).max().item()))
        score = score + must_keep.float() * 2.0  # confidence is a sigmoid in [0, 1]
    k = min(k, N)
    _, idx = torch.topk(score, k, dim=1)
    idx, _ = torch.sort(idx, dim=1)
    return idx, k


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian index remapping (N_A space -> K space)
# ─────────────────────────────────────────────────────────────────────────────


def build_inverse_index(keep_idx: Tensor, num_full: int) -> Tensor:
    """inv[b, slot] = position of slot in keep_idx[b], or -1 if pruned."""
    B, K = keep_idx.shape
    inv = keep_idx.new_full((B, num_full), -1)
    pos = torch.arange(K, device=keep_idx.device).unsqueeze(0).expand(B, -1)
    return inv.scatter(1, keep_idx, pos)


def remap_match_indices(
    indices: Sequence[Tuple[Optional[Tensor], Optional[Tensor]]],
    inv: Tensor,
):
    """Split the det-head Hungarian match into the two forms the losses need.

    kept_full : (pred_idx, target_idx) with pruned pairs dropped, still in N_A space.
                For the motion loss, which runs on scatter-back (N_A-slot) outputs.
    mapped_k  : (pred_idx in K space with -1 for pruned, target_idx), order kept.
                For the Generative Decoder loss, which relies on target order.
    """
    kept_full, mapped_k = [], []
    for b, (p, t) in enumerate(indices):
        if p is None or t is None or len(p) == 0:
            kept_full.append((p, t))
            mapped_k.append((p, t))
            continue
        pk = inv[b, p.to(inv.device)].to(p.device)
        m = pk >= 0
        kept_full.append((p[m], t[m]))
        mapped_k.append((pk, t))
    return kept_full, mapped_k


# ─────────────────────────────────────────────────────────────────────────────
# Generative Decoder: GRU initial hidden state
# ─────────────────────────────────────────────────────────────────────────────


def gru_hidden_original(hidden_states: Tensor, num_layers: int) -> Tensor:
    """Verbatim original behaviour. [B, N1, D] -> [L, B*N1, D/L] (token-mixing)."""
    return hidden_states.clone().reshape(num_layers, -1, hidden_states.shape[-1] // num_layers)


def gru_hidden_per_token(hidden_states: Tensor, num_layers: int) -> Tensor:
    """Per-token split (what the original comment `[L, B * N1, 64]` describes).

    Each token's D channels become its own L chunks of D/L. Output no longer depends
    on N1 or on other samples in the batch. Changes the model -> needs fine-tuning.
    """
    B, N1, D = hidden_states.shape
    return (
        hidden_states.reshape(B * N1, num_layers, D // num_layers)
        .permute(1, 0, 2)
        .contiguous()
    )


def gru_hidden_preserve(
    hidden_pruned: Tensor,
    keep_idx: Tensor,
    fallback_agent: Tensor,
    num_layers: int,
) -> Tensor:
    """Reproduce the ORIGINAL h0 for the kept rows while running on K + N_E tokens.

    Rebuilds the full [B, N_A + N_E, D] layout (kept slots from the pruned tensor,
    pruned slots from `fallback_agent`), applies the original reshape, then picks the
    rows that belong to the kept tokens. Rows whose sources were all kept (ensure it
    with gru_source_mask) get bit-identical h0; others read fallback values.

    hidden_pruned:  [B, K + N_E, D]  (agents in keep_idx order, then ego)
    fallback_agent: [B, N_A, D]      (e.g. the pre-decoder instance feature)
    returns:        [L, B * (K + N_E), D / L]
    """
    B, KE, D = hidden_pruned.shape
    K = keep_idx.shape[1]
    num_agent_full = fallback_agent.shape[1]
    num_ego = KE - K
    agent_full = fallback_agent.to(hidden_pruned.dtype).scatter(
        1, _expand_index(keep_idx, hidden_pruned[:, :K]), hidden_pruned[:, :K]
    )
    full = torch.cat([agent_full, hidden_pruned[:, K:]], dim=1)  # [B, N1f, D]
    h_full = full.reshape(num_layers, -1, D // num_layers)
    n1f = num_agent_full + num_ego
    slots = append_ego_slots(keep_idx, num_agent_full, num_ego)  # [B, K + N_E]
    rows = (slots + torch.arange(B, device=slots.device).unsqueeze(1) * n1f).reshape(-1)
    return h_full[:, rows]


# ─────────────────────────────────────────────────────────────────────────────
# Generative Decoder loss (Eq. 6) with pruned instances
# ─────────────────────────────────────────────────────────────────────────────


def probabilistic_loss_pruned(
    output: dict,
    mapped_indices: Sequence[Tuple[Optional[Tensor], Optional[Tensor]]],
    gt_idx_bs: Sequence[Tensor],
    ego_mode_num: int,
    loss_weight: float,
) -> Tensor:
    """Same math as losses/generation_loss.py::ProbabilisticLoss(only_valid_agent=True),
    except GT agents whose matched instance was pruned (index -1) are skipped instead
    of breaking the future-slot alignment. Identical result when nothing is pruned.
    """
    present_mu = output["present_mu"]
    present_log_sigma = output["present_log_sigma"]
    future_mu = output["future_mu"]
    future_log_sigma = output["future_log_sigma"]
    var_future = torch.exp(2 * future_log_sigma)
    var_present = torch.exp(2 * present_log_sigma)

    BS, N1, C = present_mu.shape
    mask = torch.zeros((BS, N1), dtype=torch.bool, device=present_mu.device)
    pm, pls, vp = (present_mu.new_zeros((BS, N1, C)) for _ in range(3))
    fm, fls, vf = (present_mu.new_zeros((BS, N1, C)) for _ in range(3))
    ego = list(range(N1 - ego_mode_num, N1))

    for b, (p, t) in enumerate(mapped_indices):
        if t is None or len(t) == 0:
            inst, slot = list(ego), list(ego)
        else:
            inst, slot = [], []
            for i, agent in enumerate(gt_idx_bs[b]):
                hit = torch.where(t == agent)[0]
                if len(hit) == 0:
                    continue
                k = int(p[hit[0]])
                if k < 0:
                    continue
                inst.append(k)
                slot.append(i)
            inst, slot = inst + ego, slot + ego
        mask[b, slot] = True
        pm[b, slot] = present_mu[b, inst]
        pls[b, slot] = present_log_sigma[b, inst]
        vp[b, slot] = var_present[b, inst]
        fm[b, slot] = future_mu[b, slot]
        fls[b, slot] = future_log_sigma[b, slot]
        vf[b, slot] = var_future[b, slot]

    kl = pls - fls - 0.5 + (vf + (fm - pm) ** 2) / (2 * vp)
    if mask.sum() == 0:
        return present_mu.sum() * 0.0
    return torch.mean(torch.sum(kl[mask], dim=-1)) * loss_weight
