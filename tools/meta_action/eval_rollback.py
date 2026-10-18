"""Rollback analysis on GPU evaluation results (run after tools/test.py --out).

    python tools/meta_action/eval_rollback.py CONFIG \
        --off work_dirs/run_2-1.pkl --lang work_dirs/run_2-2.pkl \
        --pool work_dirs/run_2-3.pkl --collision work_dirs/run_2-4.pkl \
        [--md work_dirs/rollback_report.md]

Runs (same config / checkpoint / data, saved with tools/test.py --out):
    --off        language_module.enabled=False                       (1st pass only)
    --lang       enabled=True, counterfactual_check=False             (language, no rollback)
    --pool       enabled=True, counterfactual_check=True, rollback_mode=pool
    --collision  enabled=True, counterfactual_check=True, rollback_mode=collision

What the official planning_eval does not show:
    (A) rollback reasons: no 2nd pass / kept 2nd / rolled back for safety / 1st for its score / both collide
    (B) L2 and collision rate on sample subsets (all, samples that ran the 2nd pass, rolled-back samples),
        computed exactly like planning_eval (same PlanningMetric, same skipped samples)
    (C) 1st vs 2nd pass on samples that ran the 2nd pass, by pairing --off and --lang:
        which one is closer to the ground truth, does the higher score mean closer (pool's assumption),
        how far apart the two trajectories are

results[i] is matched with dataset[i], the same assumption planning_eval makes.
"""
import argparse
import collections
import importlib
import os
import pickle

import torch

REASONS = ("no 2nd pass", "kept 2nd", "rolled back for safety", "1st for its score", "both collide")


# ---------------------------------------------------------------- loading
def load_results(path):
    """tools/test.py --out pkl -> list of per-sample dicts (the 'img_bbox' part)."""
    with open(path, "rb") as f:
        results = pickle.load(f)
    return [r["img_bbox"] if "img_bbox" in r else r for r in results]


def load_gt(config_path):
    """Ground truth per sample, read the same way as planning_eval (None = incomplete gt, skipped there)."""
    from mmcv import Config
    from mmdet.datasets import build_dataloader, build_dataset

    cfg = Config.fromfile(config_path)
    if cfg.get("plugin", False):  # register the plugin modules, same as tools/test.py
        plugin_dir = cfg.get("plugin_dir", os.path.dirname(config_path) + "/")
        importlib.import_module(".".join(os.path.dirname(plugin_dir).split("/")))
    dataset = build_dataset(cfg.eval_config)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)
    gts = []
    for data in loader:
        mask = data["gt_ego_fut_masks"].unsqueeze(-1).repeat(1, 1, 2)
        if not mask.all():
            gts.append(None)
            continue
        gts.append(dict(
            traj=data["gt_ego_fut_trajs"].cumsum(dim=-2)[0, :6, :2],
            mask=mask[0, :6, :2],
            cmd=int(data["gt_ego_fut_cmd"].argmax(dim=-1).item()),
            fut_boxes=data["fut_boxes"],
        ))
    return gts


# ---------------------------------------------------------------- (A) reasons
def reason_of(r):
    """Why the final trajectory was chosen (see safety_check.choose). None = no rollback record."""
    if "rollback" not in r:
        return None
    if not r["second_pass"]:
        return "no 2nd pass"
    if r["second_collided"] and r["first_collided"]:
        return "both collide"
    if not r["rollback"]:
        return "kept 2nd"
    return "rolled back for safety" if r["second_collided"] else "1st for its score"


def count_reasons(results):
    count = collections.Counter(reason_of(r) for r in results)
    return {k: count.get(k, 0) for k in REASONS}


# ---------------------------------------------------------------- (B) metrics on subsets
def summarize(per_step):
    """planning_eval's numbers: cumulative mean over steps, then the average of 1s / 2s / 3s."""
    values = per_step.tolist()
    cum = [sum(values[: i + 1]) / (i + 1) for i in range(len(values))]
    return (cum[1] + cum[3] + cum[5]) / 3


def planning_metrics(results, gts, indices):
    """L2 / collision exactly like planning_eval, but only over `indices` (samples with incomplete gt skipped)."""
    from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric

    metric = PlanningMetric()
    used = 0
    for i in indices:
        gt = gts[i]
        if gt is None:
            continue
        pred = results[i]["final_planning"].unsqueeze(0)[:, :6, :2].clone()  # update() edits in place
        metric.update(pred, gt["traj"].unsqueeze(0).clone(), gt["mask"].unsqueeze(0), gt["fut_boxes"], gt["cmd"])
        used += 1
    if used == 0:
        return dict(n=0)
    out = metric.compute()
    return dict(n=used, L2=summarize(out["L2"]), obj_col=summarize(out["obj_col"]),
                obj_box_col=summarize(out["obj_box_col"]))


# ---------------------------------------------------------------- (C) 1st vs 2nd pass
def picked_score(r, cmd):
    """Score of the final trajectory: find it among the candidates of the command (planning / planning_score)."""
    cands = r["planning"][cmd]  # [M, T, 2]
    gap = (cands[:, : r["final_planning"].shape[0]] - r["final_planning"]).abs().flatten(1).sum(-1)
    idx = int(gap.argmin())
    return float(r["planning_score"][cmd, idx]), float(gap[idx])


def mean_l2(traj, gt):
    return float(torch.linalg.norm(traj[:6, :2] - gt["traj"], dim=-1).mean())


def compare_passes(off, lang, flags, gts):
    """Samples that ran the 2nd pass (flags[i]['second_pass']): off = 1st pass pick, lang = 2nd pass pick."""
    rows = []
    for i, (a, b, f, gt) in enumerate(zip(off, lang, flags, gts)):
        if gt is None or not f.get("second_pass", False):
            continue
        s1, miss1 = picked_score(a, gt["cmd"])
        s2, miss2 = picked_score(b, gt["cmd"])
        rows.append(dict(
            index=i,
            l2_first=mean_l2(a["final_planning"], gt),
            l2_second=mean_l2(b["final_planning"], gt),
            score_first=s1, score_second=s2,
            apart=float(torch.linalg.norm(a["final_planning"][:6, :2] - b["final_planning"][:6, :2], dim=-1).mean()),
            match_gap=max(miss1, miss2),  # ~0 if the final trajectory was found among the candidates
        ))
    if not rows:
        return dict(n=0, rows=rows)
    n = len(rows)
    second_closer = sum(r["l2_second"] < r["l2_first"] for r in rows)
    decided = [r for r in rows if r["score_first"] != r["score_second"] and r["l2_first"] != r["l2_second"]]
    agree = sum((r["score_second"] > r["score_first"]) == (r["l2_second"] < r["l2_first"]) for r in decided)
    apart = sorted(r["apart"] for r in rows)
    return dict(
        n=n,
        second_closer=second_closer / n,
        l2_first=sum(r["l2_first"] for r in rows) / n,
        l2_second=sum(r["l2_second"] for r in rows) / n,
        score_agrees=agree / len(decided) if decided else float("nan"),
        apart_median=apart[n // 2],
        apart_p90=apart[min(n - 1, int(0.9 * n))],
        max_match_gap=max(r["match_gap"] for r in rows),
        rows=rows,
    )


# ---------------------------------------------------------------- report
def analyze(runs, gts):
    """runs: dict name -> results (any of off / lang / pool / collision). Returns the report dict."""
    sizes = {k: len(v) for k, v in runs.items()}
    assert len(set(sizes.values())) == 1 and len(gts) in sizes.values(), f"different sample counts {sizes}, gt {len(gts)}"
    flags = runs.get("pool") or runs.get("collision")  # which samples ran the 2nd pass
    second = [i for i, f in enumerate(flags) if f.get("second_pass")] if flags else []
    report = dict(reasons={}, metrics={}, compare=None)
    for name in ("pool", "collision"):
        if name in runs:
            report["reasons"][name] = count_reasons(runs[name])
    for name, res in runs.items():
        report["metrics"][name] = {"all": planning_metrics(res, gts, range(len(res)))}
        if flags:
            report["metrics"][name]["2nd pass samples"] = planning_metrics(res, gts, second)
        if name in ("pool", "collision"):
            back = [i for i, r in enumerate(res) if r.get("rollback")]
            report["metrics"][name]["rolled-back samples"] = planning_metrics(res, gts, back)
    if "off" in runs and "lang" in runs and flags:
        report["compare"] = compare_passes(runs["off"], runs["lang"], flags, gts)
    return report


def to_markdown(report):
    lines = ["# Rollback analysis", "", "## (A) rollback reasons", "",
             "| run | " + " | ".join(REASONS) + " |", "|---|" + "---|" * len(REASONS)]
    for name, count in report["reasons"].items():
        lines.append(f"| {name} | " + " | ".join(str(count[k]) for k in REASONS) + " |")
    lines += ["", "## (B) L2 / collision (planning_eval numbers, avg of 1s/2s/3s)", "",
              "| run | samples | n | L2 (m) | obj_col (%) | obj_box_col (%) |", "|---|---|---|---|---|---|"]
    for name, groups in report["metrics"].items():
        for group, m in groups.items():
            if m["n"] == 0:
                lines.append(f"| {name} | {group} | 0 | - | - | - |")
            else:
                lines.append(f"| {name} | {group} | {m['n']} | {m['L2']:.4f} | {100 * m['obj_col']:.3f} | "
                             f"{100 * m['obj_box_col']:.3f} |")
    c = report["compare"]
    lines += ["", "## (C) 1st vs 2nd pass (samples that ran the 2nd pass, --off vs --lang)", ""]
    if not c or c["n"] == 0:
        lines.append("not available (needs --off, --lang and --pool or --collision)")
    else:
        lines += [
            f"- samples: {c['n']}",
            f"- 2nd pass closer to gt: {100 * c['second_closer']:.1f}%  (mean L2 1st {c['l2_first']:.4f} m, 2nd {c['l2_second']:.4f} m)",
            f"- higher score = closer to gt (pool's assumption): {100 * c['score_agrees']:.1f}%",
            f"- distance between the two picks: median {c['apart_median']:.3f} m, 90% {c['apart_p90']:.3f} m",
            f"- check: max gap finding the pick among candidates {c['max_match_gap']:.2e} (should be ~0)",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the config used for tools/test.py (its eval_config gives the ground truth)")
    for name in ("off", "lang", "pool", "collision"):
        parser.add_argument(f"--{name}", help=f"tools/test.py --out pkl of the '{name}' run")
    parser.add_argument("--md", help="also write the report to this markdown file")
    args = parser.parse_args()

    runs = {name: load_results(getattr(args, name)) for name in ("off", "lang", "pool", "collision")
            if getattr(args, name)}
    assert runs, "give at least one run pkl"
    report = analyze(runs, load_gt(args.config))
    text = to_markdown(report)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
