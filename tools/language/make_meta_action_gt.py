"""GT meta-action per sample + threshold sanity check.  CPU only, no GPU / mmcv needed.

    python tools/language/make_meta_action_gt.py \
        --pkl data/infos/nuscenes_infos_train.pkl --out data/language/meta_action_gt_train.json

What it does
  1. runs the real MetaActionHead rule on every sample's GT ego future trajectory (gt_ego_fut_trajs)
  2. prints the 12-class histogram   -> this is README check #2 (are 2.0m / +-20% / 0.5m/s reasonable?)
  3. compares the rule's lateral class with gt_ego_fut_cmd (they use the same rule, so ~100% is expected;
     a low number means --traj-mode is wrong)
  4. writes {sample_token: meta_action_id} for tools/language/gen_reason_labels.py and training

ASSUMPTIONS to verify on the first run (printed at the top):
  * infos is a pkl with a list under key 'infos' (or the list itself), each item has 'token',
    'gt_ego_fut_trajs' [T,2], 'gt_ego_fut_masks' [T], 'gt_ego_fut_cmd' [3]
  * gt_ego_fut_trajs holds per-step OFFSETS like plan_reg (--traj-mode delta). If your pkl stores positions use
    --traj-mode position.
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import load_meta_action_module  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-mode", choices=["delta", "position"], default="delta")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--lat-thresh", type=float, default=2.0)
    ap.add_argument("--acc-ratio", type=float, default=1.2)
    ap.add_argument("--dec-ratio", type=float, default=0.8)
    ap.add_argument("--stop-speed", type=float, default=0.5)
    ap.add_argument("--num-speed-steps", type=int, default=2)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    ma = load_meta_action_module(args.repo_root)
    head = ma.MetaActionHead(
        ego_fut_ts=6, ego_fut_mode=6, dt=args.dt, lat_thresh=args.lat_thresh, acc_ratio=args.acc_ratio,
        dec_ratio=args.dec_ratio, stop_speed=args.stop_speed, num_speed_steps=args.num_speed_steps,
    )

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    print(f"[info] {len(infos)} samples; keys of first item: {sorted(infos[0].keys())[:12]} ...")

    tokens, trajs, cmds = [], [], []
    n_invalid = 0
    for info in infos:
        masks = np.asarray(info["gt_ego_fut_masks"])
        if masks.min() < 1:  # not enough future frames
            n_invalid += 1
            continue
        tokens.append(info["token"])
        trajs.append(np.asarray(info["gt_ego_fut_trajs"], dtype=np.float32))
        cmds.append(np.asarray(info["gt_ego_fut_cmd"]))
    trajs = torch.from_numpy(np.stack(trajs))  # [N, T, 2]
    cmds = torch.from_numpy(np.stack(cmds)).float()
    if args.traj_mode == "position":
        trajs = torch.cat([trajs[:, :1], trajs[:, 1:] - trajs[:, :-1]], dim=1)
    print(f"[info] valid {len(tokens)}  skipped (short future) {n_invalid}")
    cum = trajs.cumsum(1)
    print(f"[sanity] mean final position  x(lateral)={cum[:, -1, 0].mean():.2f}m  y(forward)={cum[:, -1, 1].mean():.2f}m"
          "  (forward should be clearly positive and larger than |lateral|)")

    metas, lats = [], []
    for s in range(0, len(tokens), args.chunk):
        plan_reg = trajs[s : s + args.chunk][:, None, None]  # [B, 1, 1, T, 2]
        out = head(plan_reg)
        metas.append(out["meta_action"][:, 0, 0])
        lats.append(out["lateral"][:, 0, 0])
    metas, lats = torch.cat(metas), torch.cat(lats)

    agree = (lats == cmds.argmax(-1)).float().mean().item()
    print(f"[sanity] lateral class == gt_ego_fut_cmd : {agree * 100:.1f}%   (expect ~100%)")
    if agree < 0.95:
        print("  !! low agreement -> check --traj-mode (delta vs position) and the x/y axis assumption")

    counts = Counter(metas.tolist())
    total = len(metas)
    print("\n  id  name                    count    share")
    for i in range(ma.NUM_META_ACTIONS):
        name = ma.MetaActionHead.to_text(i)
        print(f"  {i:2d}  {name:<22s} {counts.get(i, 0):6d}  {counts.get(i, 0) / total * 100:6.2f}%")
    top = max(counts.values()) / total
    print(f"\n  top class share {top * 100:.1f}%  |  empty classes: {[i for i in range(12) if counts.get(i, 0) == 0]}")
    if top > 0.6:
        print("  !! one class > 60%: language data will be dominated by it (consider re-tuning thresholds or class-balanced sampling)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({t: int(m) for t, m in zip(tokens, metas.tolist())}, f)
    with open(args.out + ".hist.json", "w") as f:
        json.dump({ma.MetaActionHead.to_text(i): counts.get(i, 0) for i in range(ma.NUM_META_ACTIONS)}, f, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
