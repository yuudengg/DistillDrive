"""CPU check of MetaActionHead with synthetic trajectories.

    python tools/meta_action/test_meta_action.py
"""
import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from synthetic_traj import make_batch, make_single_traj # noqa: E402

# meta_action.py only needs BaseModule + PLUGIN_LAYERS from mmcv; stub them if mmcv is missing
try:
    import mmcv # noqa: F401
except ImportError:
    class _Registry:
        def register_module(self, *args, **kwargs):
            return lambda cls: cls

    for name in ["mmcv", "mmcv.runner", "mmcv.runner.base_module",
                 "mmcv.cnn", "mmcv.cnn.bricks", "mmcv.cnn.bricks.registry"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mmcv.runner.base_module"].BaseModule = torch.nn.Module
    sys.modules["mmcv.cnn.bricks.registry"].PLUGIN_LAYERS = _Registry()

# load the file directly, the package __init__ pulls in the whole mmdet3d stack
path = os.path.join(ROOT, "projects", "mmdet3d_plugin", "models", "motion", "meta_action.py")
spec = importlib.util.spec_from_file_location("meta_action", path)
meta_action = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta_action)

head = meta_action.MetaActionHead()
to_text = head.to_text
ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print("OK  " if good else "FAIL", msg)


print("=== 1. all 12 meta-actions from the generator ===")
for lateral in range(3):
    for longitudinal in range(4):
        expect = lateral * 4 + longitudinal
        plan_reg = make_single_traj(lateral, longitudinal)[None, None, None] # [1, 1, 1, T, 2]
        got = int(head(plan_reg)["meta_action"][0, 0, 0])
        check(got == expect, f"{to_text(expect):22s} -> {to_text(got)}")

print("\n=== 2. stop + turn is physically consistent ===")
for lateral in (0, 1):
    traj = make_single_traj(lateral, 3)
    final_x = float(traj.cumsum(0)[-1, 0])
    late_speed = float(traj[-2:].norm(dim=-1).mean() / 0.5)
    check(abs(final_x) >= 2.0 and late_speed < 0.5,
          f"{to_text(lateral * 4 + 3):22s} final x {final_x:+.2f} m, late speed {late_speed:.2f} m/s")

print("\n=== 3. plan_reg-shaped batch (ego_fut_mode=6, seed variations) ===")
for seed in range(5):
    plan_reg, expect = make_batch(batch_size=4, ego_fut_mode=6, seed=seed)
    got = head(plan_reg)["meta_action"]
    check(tuple(plan_reg.shape) == (4, 1, 18, 6, 2) and torch.equal(got, expect),
          f"seed {seed}: shape {tuple(plan_reg.shape)}, {int((got == expect).sum())}/{expect.numel()} match")

print("\n=== 4. same batch twice with same seed ===")
a, _ = make_batch(2, seed=7)
b, _ = make_batch(2, seed=7)
check(torch.equal(a, b), "make_batch is reproducible")

print("\n=== 5. lateral rule == dataset gt_ego_fut_cmd rule (random) ===")
torch.manual_seed(0)
plan_reg = torch.randn(64, 1, 18, 6, 2)
final_x = plan_reg.cumsum(-2)[..., -1, 0]
ref = torch.where(final_x >= 2, 0, torch.where(final_x <= -2, 1, 2))
check(torch.equal(head(plan_reg)["lateral"], ref), f"{ref.numel()} trajectories match converter rule")

print("\n=== 6. selected mode follows cmd group ===")
plan_reg, expect = make_batch(batch_size=2, seed=1)
plan_cls = torch.zeros(2, 1, 18)
plan_cls[0, 0, 1 * 6 + 4] = 5.0
plan_cls[0, 0, 0 * 6 + 0] = 9.0 # higher score but wrong cmd group -> ignored
plan_cls[1, 0, 0 * 6 + 2] = 5.0
cmd = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float) # left, right
out = head(plan_reg, plan_cls, cmd)
check(out["selected_mode"].tolist() == [10, 2]
      and out["selected_meta_action"].tolist() == [int(expect[0, 0, 10]), int(expect[1, 0, 2])],
      f"selected_mode={out['selected_mode'].tolist()}, "
      f"meta={[to_text(i) for i in out['selected_meta_action']]}")

print("\nALL PASSED" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
