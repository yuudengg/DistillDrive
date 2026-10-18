"""Rollback (counterfactual check) for the language feedback 2nd pass.

The 2nd pass (language -> adaLN -> refine) can be worse than the 1st, e.g. the
language model says "accelerate" with a car in front. Instead of trusting the
2nd pass blindly, the 1st and 2nd candidates are pooled and judged together by
the existing collision check (decoder.HierarchicalPlanningDecoder):

    pool = [2nd pass M modes | 1st pass M modes]
        ->  collision_mask (True = collides)  ->  collision_penalty (-999)  ->  choose

Modes (language_module.rollback_mode):
    "pool"      : best score among all 2M candidates (collided ones get -999).
    "collision" : keep the 2nd pass choice, fall back to the 1st pass choice
                  only if the 2nd one collides and the 1st one does not.
                  (Safer fallback if 1st / 2nd scores turn out not comparable.)

Inference only (decoder), no effect on training / loss.
"""
import torch

ROLLBACK_MODES = ("pool", "collision")
# language_module keys owned by the rollback (for a key check on the head side)
ROLLBACK_KEYS = ("counterfactual_check", "rollback_mode")


def check_rollback_cfg(lang_cfg):
    """Check the rollback config once when the head is built, so mistakes show up before training.

    lang_cfg: the language_module dict (enabled / counterfactual_check / rollback_mode ...)
    """
    mode = lang_cfg.get("rollback_mode", "pool")
    assert mode in ROLLBACK_MODES, (
        f"language_module.rollback_mode must be one of {ROLLBACK_MODES}, got {mode!r}"
    )
    if lang_cfg.get("counterfactual_check", False):
        assert lang_cfg.get("enabled", False), (
            "language_module.counterfactual_check=True needs enabled=True "
            "(the rollback compares the 1st and 2nd pass, the 2nd pass only runs when enabled)"
        )


def pool_candidates(new_cls, new_reg, old_cls, old_reg, use):
    """Concatenate 2nd (new) and 1st (old) pass candidates of the selected command.

    new_cls, old_cls: [B, M]  /  new_reg, old_reg: [B, M, T, 2]  /  use: [B] bool
    (samples with use=False did not run the 2nd pass, so new == old there)

    Returns pool_cls [B, 2M], pool_reg [B, 2M, T, 2].
    The 1st pass half of use=False samples is blocked with -inf (it is a duplicate).
    2nd pass comes first, so on a tie argmax keeps the 2nd pass.
    """
    use = use.to(device=new_cls.device, dtype=torch.bool)
    blocked = torch.full_like(old_cls, float("-inf"))
    old_cls = torch.where(use[:, None], old_cls, blocked)
    pool_cls = torch.cat([new_cls, old_cls], dim=1)
    pool_reg = torch.cat([new_reg, old_reg], dim=1)
    return pool_cls, pool_reg


def choose(scored_cls, collided, pool_reg, use, mode="pool"):
    """Pick the final trajectory from the pooled candidates.

    scored_cls: [B, 2M] pooled scores after collision_penalty (collided candidates lowered)
    collided:   [B, 2M] bool from collision_mask (True = collides), read directly
                instead of guessing it from how much the score dropped
    pool_reg:   [B, 2M, T, 2]
    use:        [B] bool, samples that ran the 2nd pass

    Returns final [B, T, 2], from_first [B] bool (True = rolled back to 1st pass).
    """
    assert mode in ROLLBACK_MODES, f"rollback_mode must be one of {ROLLBACK_MODES}, got {mode}"
    bs, num_pool = scored_cls.shape
    num_new = num_pool // 2
    bs_indices = torch.arange(bs, device=scored_cls.device)
    use = use.to(device=scored_cls.device, dtype=torch.bool)

    if mode == "pool":
        idx = scored_cls.argmax(dim=-1)
    else:
        collided = collided.to(device=scored_cls.device, dtype=torch.bool)
        best_new = scored_cls[:, :num_new].argmax(dim=-1)
        best_old = scored_cls[:, num_new:].argmax(dim=-1) + num_new
        rollback = use & collided[bs_indices, best_new] & ~collided[bs_indices, best_old]
        idx = torch.where(rollback, best_old, best_new)

    final = pool_reg[bs_indices, idx]
    from_first = idx >= num_new
    return final, from_first
