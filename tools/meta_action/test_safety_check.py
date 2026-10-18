"""CPU check of the rollback (safety_check.py + HierarchicalPlanningDecoder) with synthetic scenes.

    python tools/meta_action/test_safety_check.py

Uses the real decoder.py collision check; only mmdet (registry) and cv2 are stubbed.
"""
import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLUGIN = os.path.join(ROOT, "projects", "mmdet3d_plugin")


# ---------- load decoder.py without the full mmdet3d stack ----------
class _Registry:
    def register_module(self, *args, **kwargs):
        return lambda cls: cls


def _stub(name, **attrs):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)
    return sys.modules[name]


try:
    import mmdet.core.bbox.builder # noqa: F401
except ImportError:
    for name in ["mmdet", "mmdet.core", "mmdet.core.bbox"]:
        _stub(name)
    _stub("mmdet.core.bbox.builder", BBOX_CODERS=_Registry())
try:
    import cv2 # noqa: F401
except ImportError:
    _stub("cv2")  # datasets/utils.py imports it, box3d_to_corners does not use it

for pkg, path in [
    ("projects", os.path.join(ROOT, "projects")),
    ("projects.mmdet3d_plugin", PLUGIN),
    ("projects.mmdet3d_plugin.core", os.path.join(PLUGIN, "core")),
    ("projects.mmdet3d_plugin.datasets", os.path.join(PLUGIN, "datasets")),
    ("projects.mmdet3d_plugin.models", os.path.join(PLUGIN, "models")),
    ("projects.mmdet3d_plugin.models.detection3d", os.path.join(PLUGIN, "models", "detection3d")),
    ("projects.mmdet3d_plugin.models.motion", os.path.join(PLUGIN, "models", "motion")),
]:
    _stub(pkg).__path__ = [path]  # namespace only, skip the heavy package __init__


def load(name, *parts):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PLUGIN, *parts))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("projects.mmdet3d_plugin.core.box3d", "core", "box3d.py")
load("projects.mmdet3d_plugin.datasets.utils", "datasets", "utils.py")
load("projects.mmdet3d_plugin.models.detection3d.decoder", "models", "detection3d", "decoder.py")
safety = load("projects.mmdet3d_plugin.models.motion.safety_check", "models", "motion", "safety_check.py")
decoder_mod = load("projects.mmdet3d_plugin.models.motion.decoder", "models", "motion", "decoder.py")

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print("OK  " if good else "FAIL", msg)


# ---------- synthetic scene ----------
# ego frame: x = lateral, y = forward (same as gt_ego_fut_trajs). 6 steps (3 s).
M, T, NA, MA = 6, 6, 3, 6  # ego modes, ego steps, agents, agent modes


def straight(dist, lateral=0.0):
    """[T, 2] per-step offsets reaching (lateral, dist) at the end (decoder does cumsum)."""
    step = torch.tensor([lateral / T, dist / T])
    return step.repeat(T, 1)


def scene(bs):
    """One parked car 10 m ahead on every sample. Other anchors are low-score (ignored)."""
    det_anchors = torch.zeros(bs, NA, 11)
    det_anchors[..., 3:6] = torch.tensor([4.08, 1.73, 1.56]).log()  # W, L, H are log-encoded
    det_anchors[..., 6] = 1.0  # sin(yaw) -> heading along +y like the ego
    det_anchors[:, 0, 1] = 10.0  # car 0 at (0, 10)
    det_anchors[:, 1:, 0] = 50.0  # others far away
    det_cls = torch.full((bs, NA, 10), -5.0)
    det_cls[:, 0, 0] = 5.0  # only car 0 is confident (> 0.5)
    motion_cls = torch.zeros(bs, NA, MA)
    motion_cls[..., 0] = 5.0
    motion_reg = torch.zeros(bs, NA, MA, 12, 2)  # parked
    det_output = {"classification": [det_cls], "prediction": [det_anchors]}
    motion_output = {"classification": [motion_cls], "prediction": [motion_reg]}
    return det_output, motion_output


def plans(bs, per_sample):
    """per_sample: list of (list of M (dist, score)) for the straight command -> cls [B,3,M], reg [B,3,M,T,2] (cumsum)."""
    cls = torch.full((bs, 3, M), -9.0)
    reg = torch.zeros(bs, 3, M, T, 2)
    for b, modes in enumerate(per_sample):
        for m, (dist, score) in enumerate(modes):
            cls[b, 2, m] = score
            reg[b, 2, m] = straight(dist)
    return cls, reg.cumsum(dim=-2)


dec = decoder_mod.HierarchicalPlanningDecoder(ego_fut_ts=T, ego_fut_mode=M, use_rescore=True)
data = {"gt_ego_fut_cmd": torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)}  # straight
GO, STOP = 15.0, 2.0  # GO passes through the car at y=10, STOP ends 2 m ahead

# sample 0: 2nd all GO (collide), 1st has STOP  -> must roll back
# sample 1: 2nd has STOP (safe), 1st all GO       -> keep 2nd
# sample 2: did not run the 2nd pass (use=False)  -> same as no rollback
# sample 3: both safe, 1st scores higher          -> "pool" takes 1st, "collision" keeps 2nd
new = [
    [(GO, 3.0)] * M,
    [(STOP, 1.0)] + [(GO, 3.0)] * (M - 1),
    [(STOP, 1.0)] + [(GO, 0.5)] * (M - 1),
    [(STOP, 1.0)] * M,
]
old = [
    [(STOP, 1.0)] + [(GO, 2.0)] * (M - 1),
    [(GO, 2.0)] * M,
    new[2],
    [(STOP, 2.0)] * M,
]
use = torch.tensor([True, True, False, True])
new_cls, new_reg = plans(4, new)
old_cls, old_reg = plans(4, old)
det_output, motion_output = scene(4)
end_y = lambda traj: round(float(traj[-1, 1]), 3)

# ---------- 1. safety_check functions alone ----------
print("== 1. pool_candidates / choose ==")
pc, pr = safety.pool_candidates(new_cls[:, 2], new_reg[:, 2], old_cls[:, 2], old_reg[:, 2], use)
check(pc.shape == (4, 2 * M) and pr.shape == (4, 2 * M, T, 2), f"pool shape {tuple(pc.shape)} / {tuple(pr.shape)}")
check(torch.isinf(pc[2, M:]).all() and not torch.isinf(pc[[0, 1, 3], M:]).any(),
      "1st pass half blocked only for use=False samples")
col4 = torch.tensor([[False, True, False, False]])  # new mode 1 collided
scored = torch.tensor([[1.0, 2.0 - 999, 0.5, 0.0]])
reg4 = torch.arange(4.0)[None, :, None, None].repeat(1, 1, T, 2)
f, ff = safety.choose(scored, col4, reg4, torch.tensor([True]), "pool")
check(ff.tolist() == [False] and f[0, 0, 0] == 0, "pool: best non-collided overall (2nd mode 0)")
f, ff = safety.choose(scored, col4, reg4, torch.tensor([True]), "collision")
check(ff.tolist() == [False] and f[0, 0, 0] == 0, "collision: 2nd best is safe -> keep 2nd")
tie = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
_, ff = safety.choose(tie, torch.zeros(1, 4, dtype=torch.bool), reg4, torch.tensor([True]), "pool")
check(ff.tolist() == [False], "tie -> 2nd pass wins")
# collision mode reads the mask, not the score drop: a small penalty still rolls back
small = torch.tensor([[3.0 - 0.5, 0.0, 1.0, 0.0]])
_, ff = safety.choose(small, torch.tensor([[True, True, False, False]]), reg4, torch.tensor([True]), "collision")
check(ff.tolist() == [True], "collision: penalty of only 0.5 -> still detected, roll back")

# ---------- 1b. rescore split: collision_mask + collision_penalty ----------
print("== 1b. collision_mask / collision_penalty ==")
inputs = (motion_output["classification"][-1], motion_output["prediction"][-1],
          det_output["prediction"][-1], det_output["classification"][-1].sigmoid().max(dim=-1)[0])
mask = dec.collision_mask(new_reg[:, 2], *inputs)
check(mask.tolist() == [[True] * M, [False] + [True] * (M - 1), [False] + [True] * (M - 1), [False] * M],
      f"collision_mask: GO collides, STOP is safe (sample0 all {mask[0].tolist()})")
check(torch.equal(dec.rescore(new_cls[:, 2], new_reg[:, 2], *inputs),
                  dec.collision_penalty(new_cls[:, 2], mask)),
      "rescore == collision_penalty(collision_mask)")
check(torch.equal(dec.collision_penalty(new_cls[:, 2], mask)[0], new_cls[0, 2]) and mask[0].all(),
      "all collide: penalty skips them, but the mask still says True (and is not modified)")

# ---------- 2. real decoder: no rollback key == original path ----------
print("== 2. decoder without rollback (regression) ==")
cls_a, fin_a, ff_a = dec.select(det_output, motion_output, new_cls.clone(), new_reg.clone(), data)
check(ff_a is None, "no 'first' -> from_first is None")
check([end_y(fin_a[b]) for b in range(4)] == [GO, STOP, STOP, STOP],
      f"original rescore: sample0 all collide -> no rescore (GO), others avoid the car {[end_y(fin_a[b]) for b in range(4)]}")

# ---------- 3. real decoder: rollback ----------
for mode in ["pool", "collision"]:
    print(f"== 3. decoder with rollback, mode={mode} ==")
    first = dict(classification=old_cls.clone(), prediction=old_reg.clone(), use=use, mode=mode)
    cls_b, fin_b, ff_b = dec.select(det_output, motion_output, new_cls.clone(), new_reg.clone(), data, first=first)
    ys = [end_y(fin_b[b]) for b in range(4)]
    check(ff_b[0] and ys[0] == STOP, f"sample0: 2nd all collide, 1st safe -> roll back to 1st STOP (y={ys[0]})")
    check(not ff_b[1] and ys[1] == STOP, f"sample1: 2nd safe -> keep 2nd (y={ys[1]})")
    check(not ff_b[2] and torch.equal(fin_b[2], fin_a[2]), "sample2: use=False -> identical to no rollback")
    want3 = mode == "pool"
    check(bool(ff_b[3]) == want3, f"sample3: both safe, 1st higher -> {'1st' if want3 else '2nd'} ({mode})")
    check(cls_b.shape == (4, 3, M) and torch.equal(cls_b[:, :2], new_cls[:, :2]),
          "planning_score: 2nd pass scores, other commands untouched")

# ---------- 4. decode(): output key ----------
print("== 4. decode() output ==")
planning_output = {
    "classification": [new_cls.reshape(4, 1, 3 * M)],
    "prediction": [torch.diff(new_reg, dim=-2, prepend=torch.zeros(4, 3, M, 1, 2)).reshape(4, 1, 3 * M, T, 2)],
    "anchor_queue": [torch.zeros(4, 1, 11)],
    "period": torch.zeros(4, 1),
}
out = dec.decode(det_output, motion_output, planning_output, data)
check("rollback" not in out[0], "without rollback info -> no 'rollback' key (same output as before)")
planning_output["rollback"] = dict(
    classification=old_cls.reshape(4, 1, 3 * M),
    prediction=torch.diff(old_reg, dim=-2, prepend=torch.zeros(4, 3, M, 1, 2)).reshape(4, 1, 3 * M, T, 2),
    use=use,
    mode="pool",
)
out = dec.decode(det_output, motion_output, planning_output, data)
check([o["rollback"] for o in out] == [True, False, False, True], f"rollback flags {[o['rollback'] for o in out]}")
check(abs(float(out[0]["final_planning"][-1, 1]) - STOP) < 1e-4, "sample0 final_planning is the 1st pass STOP")

# ---------- 5. check_rollback_cfg: config mistakes fail when the head is built ----------
print("== 5. check_rollback_cfg ==")


def fails(cfg):
    try:
        safety.check_rollback_cfg(cfg)
    except AssertionError:
        return True
    return False


check(not fails({}), "empty language_module (all defaults) -> ok")
check(not fails(dict(enabled=False, counterfactual_check=False, rollback_mode="pool")),
      "default config of language_config.py -> ok")
for mode in safety.ROLLBACK_MODES:
    check(not fails(dict(enabled=True, counterfactual_check=True, rollback_mode=mode)), f"rollback on, mode={mode} -> ok")
check(fails(dict(rollback_mode="colision")), "value typo 'colision' -> error (even with rollback off)")
check(fails(dict(enabled=True, counterfactual_check=True, rollback_mode="Pool")), "value typo 'Pool' -> error")
check(fails(dict(enabled=False, counterfactual_check=True)), "counterfactual_check=True but enabled=False -> error")
check(set(safety.ROLLBACK_KEYS) == {"counterfactual_check", "rollback_mode"}, "ROLLBACK_KEYS")

print("\nALL OK" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
