"""CPU check of AdaLNZero (feedback condition) with synthetic trajectories.

    python tools/meta_action/test_adaln.py
"""
import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from synthetic_traj import make_batch # noqa: E402

# adaln.py / meta_action.py only need BaseModule + PLUGIN_LAYERS from mmcv; stub them if mmcv is missing
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


def load(name):
    # load the file directly, the package __init__ pulls in the whole mmdet3d stack
    path = os.path.join(ROOT, "projects", "mmdet3d_plugin", "models", "motion", name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AdaLNZero = load("adaln").AdaLNZero
MetaActionHead = load("meta_action").MetaActionHead

B, N, D = 4, 18, 256
torch.manual_seed(0)
feature = torch.randn(B, N, D)
meta_idx = torch.randint(0, 12, (B,))
ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print("OK  " if good else "FAIL", msg)


def run(adaln, meta_idx, **kwargs):
    gamma, beta, alpha = adaln(meta_idx, **kwargs)
    return adaln.modulate(feature, gamma, beta, alpha)


print("=== 1. zero init: output == input ===")
adaln = AdaLNZero(D)
gamma, beta, alpha = adaln(meta_idx)
check(gamma.shape == beta.shape == alpha.shape == (B, 1, D), f"[B] -> {tuple(alpha.shape)}")
check(torch.equal(adaln.modulate(feature, gamma, beta, alpha), feature), "plain")
adaln_opt = AdaLNZero(D, use_uncertainty=True, extra_dim=32)
out = run(adaln_opt, meta_idx, uncertainty=torch.rand(B), extra=torch.randn(B, 32))
check(torch.equal(out, feature), "with uncertainty + extra")

print("\n=== 2. zero init still learns (gradient reaches alpha) ===")
adaln = AdaLNZero(D)
target = feature + 0.5 * adaln.norm(feature) # reachable with alpha = 0.5
((run(adaln, meta_idx) - target) ** 2).mean().backward()
grad = adaln.modulation[-1].bias.grad
check(grad[2 * D:].abs().sum() > 0, f"alpha bias grad {grad[2 * D:].abs().sum():.2e} > 0")
optim = torch.optim.Adam(adaln.parameters(), lr=0.01)
losses = []
for _ in range(50):
    optim.zero_grad()
    loss = ((run(adaln, meta_idx) - target) ** 2).mean()
    loss.backward()
    optim.step()
    losses.append(float(loss))
check(losses[-1] < losses[0] * 0.5, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
gamma_grad = adaln.modulation[-1].weight.grad[:D].abs().sum()
check(gamma_grad > 0, f"gamma weight grad {gamma_grad:.2e} > 0 after alpha moved")
print("     gate_stats:", adaln.gate_stats())

print("\n=== 3. non-zero weights: output changes, depends on meta_idx ===")
adaln = AdaLNZero(D)
torch.nn.init.normal_(adaln.modulation[-1].weight, std=0.1)
diff = (run(adaln, meta_idx) - feature).abs().mean()
check(diff > 0, f"|out - in| = {diff:.4f}")
out_a = run(adaln, torch.zeros(B, dtype=torch.long))
out_b = run(adaln, torch.full((B,), 6, dtype=torch.long))
check(not torch.allclose(out_a, out_b), "meta 0 vs meta 6 give different outputs")

print("\n=== 4. uncertainty: higher -> larger gate ===")
adaln = AdaLNZero(D, use_uncertainty=True)
torch.nn.init.constant_(adaln.uncertainty_proj.weight, 0.5)
_, _, alpha_low = adaln(meta_idx, uncertainty=torch.full((B,), 0.1))
_, _, alpha_high = adaln(meta_idx, uncertainty=torch.full((B,), 0.9))
check(alpha_high.abs().mean() > alpha_low.abs().mean(),
      f"alpha {alpha_low.abs().mean():.3f} (u=0.1) < {alpha_high.abs().mean():.3f} (u=0.9)")
_, _, alpha_none = adaln(meta_idx)
check(torch.equal(alpha_none, torch.zeros_like(alpha_none)), "uncertainty=None -> no push")

print("\n=== 5. options ===")
_, _, alpha = AdaLNZero(D, gate_init=0.01)(meta_idx)
check(torch.allclose(alpha, torch.full_like(alpha, 0.01)), "gate_init=0.01 -> alpha starts at 0.01")
per_mode = torch.randint(0, 12, (B, N))
gamma, beta, alpha = AdaLNZero(D)(per_mode)
check(alpha.shape == (B, N, D), f"per-mode meta_idx [B, N] -> {tuple(alpha.shape)}")

print("\n=== 6. MetaActionHead -> AdaLNZero on synthetic batch ===")
plan_reg, _ = make_batch(B, seed=1) # [B, 1, 18, 6, 2]
plan_cls = torch.randn(B, 1, N)
cmd = torch.nn.functional.one_hot(torch.randint(0, 3, (B,)), 3)
selected = MetaActionHead()(plan_reg, plan_cls, cmd)["selected_meta_action"] # [B]
adaln = AdaLNZero(D)
gamma, beta, alpha = adaln(selected)
check(torch.equal(adaln.modulate(feature, gamma, beta, alpha), feature),
      f"selected_meta_action {selected.tolist()} -> zero-init output == input")

print("\n=== 7. int meta_idx (one for the whole batch) ===")
adaln = AdaLNZero(D)
torch.nn.init.normal_(adaln.modulation[-1].weight, std=0.1)
gamma, beta, alpha = adaln(5)
check(alpha.shape == (1, 1, D), f"int -> {tuple(alpha.shape)}")
check(torch.equal(adaln(torch.tensor(5))[2], alpha), "0-dim tensor == int")
check(torch.equal(adaln(torch.tensor([5]))[2], alpha), "int 5 == [B] tensor [5]")
check(adaln.modulate(feature, gamma, beta, alpha).shape == feature.shape, "int cond broadcasts over [B, N, D]")
check(AdaLNZero(D, num_meta=12).meta_embed.num_embeddings == 12, "config name num_meta (same as A's language_config)")

print("\n=== 8. real AdaLNZero in A's feedback_refine ===")
sys.path.insert(0, os.path.join(ROOT, "projects", "mmdet3d_plugin", "models", "motion", "language"))
from fake_refine_head import FakeRefineHead # noqa: E402
from feedback_integration import MockLanguageModule, feedback_refine # noqa: E402

head = FakeRefineHead(embed_dims=32, num_layers=6, seed=0).eval()
feat = torch.randn(4, N, 32)
base_reg, _ = head.run_whole(feat)
adaln = AdaLNZero(32)
reg, _, info = feedback_refine(head, feat, use_feedback=True, split=3, lm=MockLanguageModule(fixed_meta=5),
                               adaln=adaln, threshold=1.0)
check(info["triggered"] and torch.allclose(reg, base_reg, atol=1e-6), "zero init -> same as 1st pass")
reg.pow(2).mean().backward()
check(adaln.modulation[-1].bias.grad.abs().sum() > 0, "zero init -> gradient reaches adaLN")
torch.nn.init.normal_(adaln.modulation[-1].weight, std=0.1)
reg, _, info = feedback_refine(head, feat, use_feedback=True, split=3, lm=MockLanguageModule(fixed_meta=5),
                               adaln=adaln, scene_inputs=[{"fail": True}, None, None, None], threshold=1.0)
check(torch.allclose(reg[0], base_reg[0], atol=1e-6) and not torch.allclose(reg[1:], base_reg[1:], atol=1e-4),
      f"LM failed on sample 0 -> only sample 0 keeps 1st pass (used {info['used']})")

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
