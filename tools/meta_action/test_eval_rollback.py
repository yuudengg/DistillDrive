"""CPU check of tools/meta_action/eval_rollback.py with fake results / ground truth.

    python tools/meta_action/test_eval_rollback.py

Loads the official planning_eval (packages missing on CPU are stubbed: shapely, mmcv, mmdet, tqdm,
prettytable, cv2) and checks that eval_rollback's numbers on "all samples" are exactly the official ones.
"""
import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLUGIN = os.path.join(ROOT, "projects", "mmdet3d_plugin")


def _stub(name, **attrs):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)
    return sys.modules[name]


class _Polygon:
    """Stand-in for shapely Polygon: axis-aligned bounding boxes (enough to compare two callers of the same code)."""

    def __init__(self, pts):
        xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
        self.box = (min(xs), min(ys), max(xs), max(ys))

    def intersects(self, other):
        a, b = self.box, other.box
        return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


class _Table:
    def __init__(self):
        self.field_names, self.rows = [], []

    def add_row(self, row):
        self.rows.append(row)

    def __str__(self):
        return "\n".join(str(r) for r in self.rows)


FAKE_DATA = []  # what the stubbed dataloader yields to the official planning_eval
for name in ["shapely", "mmcv", "mmdet"]:
    _stub(name)
_stub("shapely.geometry", Polygon=_Polygon)
_stub("mmcv.utils", print_log=lambda *a, **k: None)
_stub("mmdet.datasets", build_dataset=lambda cfg: None, build_dataloader=lambda *a, **k: FAKE_DATA)
_stub("tqdm", tqdm=lambda x: x)
_stub("prettytable", PrettyTable=_Table)
try:
    import cv2  # noqa: F401
except ImportError:
    _stub("cv2")
for pkg, path in [
    ("projects", os.path.join(ROOT, "projects")),
    ("projects.mmdet3d_plugin", PLUGIN),
    ("projects.mmdet3d_plugin.datasets", os.path.join(PLUGIN, "datasets")),
    ("projects.mmdet3d_plugin.datasets.evaluation", os.path.join(PLUGIN, "datasets", "evaluation")),
    ("projects.mmdet3d_plugin.datasets.evaluation.planning", os.path.join(PLUGIN, "datasets", "evaluation", "planning")),
]:
    _stub(pkg).__path__ = [path]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("projects.mmdet3d_plugin.datasets.utils", os.path.join(PLUGIN, "datasets", "utils.py"))
official = load("projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval",
                os.path.join(PLUGIN, "datasets", "evaluation", "planning", "planning_eval.py"))
ev = load("eval_rollback", os.path.join(ROOT, "tools", "meta_action", "eval_rollback.py"))

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print("OK  " if good else "FAIL", msg)


# ---------- fake scene: N samples, ground truth driving straight, some agents ahead ----------
N, M, T = 40, 6, 6
g = torch.Generator().manual_seed(0)


def straight(speed, lateral=0.0):
    t = torch.arange(1, T + 1, dtype=torch.float32)[:, None]
    return torch.cat([torch.full_like(t, lateral) * t / T, speed * t], dim=-1)  # [T, 2] cumulative


gts, FAKE_DATA[:] = [], []
for i in range(N):
    speed = 1.0 + 2.0 * float(torch.rand(1, generator=g))
    traj = straight(speed)
    cmd = i % 3
    complete = i % 10 != 9  # every 10th sample has incomplete gt -> skipped by planning_eval
    boxes = []
    for t in range(T):  # a car parked 8 m ahead in some samples -> collisions for fast candidates
        b = torch.zeros(1, 1 if i % 4 == 0 else 0, 7)
        if i % 4 == 0:
            b[0, 0] = torch.tensor([0.0, 8.0, 0.0, 4.0, 1.8, 1.5, 1.5708])
        boxes.append(b)
    mask = torch.ones(1, T) if complete else torch.cat([torch.ones(1, T - 1), torch.zeros(1, 1)], dim=-1)
    FAKE_DATA.append({
        "gt_ego_fut_trajs": torch.diff(traj, dim=0, prepend=torch.zeros(1, 2))[None],
        "gt_ego_fut_masks": mask,
        "gt_ego_fut_cmd": torch.nn.functional.one_hot(torch.tensor([cmd]), 3).float(),
        "fut_boxes": boxes,
    })
    m = mask.unsqueeze(-1).repeat(1, 1, 2)
    gts.append(dict(traj=traj, mask=m[0], cmd=cmd, fut_boxes=boxes) if m.all() else None)


def run(seed, record=None, second=None):
    """Fake results: candidates per command, a picked index, optional rollback record."""
    rg = torch.Generator().manual_seed(seed)
    res = []
    for i in range(N):
        cands = torch.stack([torch.stack([straight(0.5 + 3 * float(torch.rand(1, generator=rg)),
                                                   float(torch.randn(1, generator=rg)))
                                          for _ in range(M)]) for _ in range(3)])  # [3, M, T, 2]
        score = torch.rand(3, M, generator=rg)
        cmd = gts[i]["cmd"] if gts[i] else 0
        idx = int(score[cmd].argmax())
        r = {"planning": cands, "planning_score": score, "final_planning": cands[cmd, idx].clone()}
        if record is not None:
            r.update(record(i))
        res.append(r)
    return res


def rec(i):  # cycle through the five reasons
    k = i % 5
    return [dict(rollback=False, second_pass=False, second_collided=False, first_collided=False),
            dict(rollback=False, second_pass=True, second_collided=False, first_collided=True),
            dict(rollback=True, second_pass=True, second_collided=True, first_collided=False),
            dict(rollback=True, second_pass=True, second_collided=False, first_collided=False),
            dict(rollback=True, second_pass=True, second_collided=True, first_collided=True)][k]


off, lang, pool = run(1), run(2), run(3, rec)

# ---------- 1. same numbers as the official planning_eval ----------
print("== 1. eval_rollback == official planning_eval on all samples ==")
for name, res in [("off", off), ("pool", pool)]:
    # copies: the official PlanningMetric.update() flips x of final_planning in place (harmless in tools/test.py,
    # which dumps the pkl before evaluating; eval_rollback reads the pkl and passes clones)
    wrapped = [{"img_bbox": {k: v.clone() if torch.is_tensor(v) else v for k, v in r.items()}} for r in res]
    off_dict = official.planning_eval(wrapped, None, None)
    mine = ev.planning_metrics(res, gts, range(N))
    same = all(abs(off_dict[k] - mine[k]) < 1e-9 for k in ["L2", "obj_col", "obj_box_col"])
    check(same and mine["n"] == sum(g is not None for g in gts),
          f"{name}: L2 {mine['L2']:.4f} == {off_dict['L2']:.4f}, obj_box_col {mine['obj_box_col']:.4f} == "
          f"{off_dict['obj_box_col']:.4f}, n {mine['n']}")
check(any(official.PlanningMetric().evaluate_single_coll(r["final_planning"][:, :2].clone(), g["fut_boxes"]).any()
          for r, g in zip(off, gts) if g is not None), "the fake scene really has collisions (collision path exercised)")
before = off[0]["final_planning"].clone()
ev.planning_metrics(off, gts, range(N))
check(torch.equal(off[0]["final_planning"], before), "planning_metrics does not modify the results (update() edits in place)")

# ---------- 2. subsets ----------
print("== 2. subsets ==")
sub = [i for i in range(N) if i % 2 == 0]
m_sub = ev.planning_metrics(off, gts, sub)
check(m_sub["n"] == sum(gts[i] is not None for i in sub), f"subset counts only the chosen samples with complete gt (n={m_sub['n']})")
check(ev.planning_metrics(off, gts, [])["n"] == 0, "empty subset -> n=0")

# ---------- 3. reasons ----------
print("== 3. rollback reasons ==")
counts = ev.count_reasons(pool)
check(list(counts.values()) == [N // 5] * 5, f"each reason counted once per 5 samples {counts}")
check(ev.reason_of({"final_planning": 0}) is None, "no record -> None")

# ---------- 4. 1st vs 2nd pass ----------
print("== 4. compare 1st vs 2nd pass ==")
cmp_ = ev.compare_passes(off, lang, pool, gts)
rows = [i for i in range(N) if gts[i] is not None and rec(i)["second_pass"]]
check(cmp_["n"] == len(rows), f"only samples that ran the 2nd pass and have gt (n={cmp_['n']})")
check(cmp_["max_match_gap"] < 1e-6, "the final trajectory is found among the candidates (score lookup is right)")
by_hand = sum(float(torch.linalg.norm(lang[i]["final_planning"] - gts[i]["traj"], dim=-1).mean())
              < float(torch.linalg.norm(off[i]["final_planning"] - gts[i]["traj"], dim=-1).mean()) for i in rows) / len(rows)
check(abs(cmp_["second_closer"] - by_hand) < 1e-12, f"2nd-closer fraction matches a by-hand count ({by_hand:.3f})")
s_hand = [(float(lang[i]["planning_score"][gts[i]["cmd"]].max()), float(off[i]["planning_score"][gts[i]["cmd"]].max())) for i in rows]
check(all(abs(r["score_second"] - s[0]) < 1e-9 and abs(r["score_first"] - s[1]) < 1e-9 for r, s in zip(cmp_["rows"], s_hand)),
      "picked scores are the scores of the chosen candidates")

# ---------- 5. whole report ----------
print("== 5. analyze + markdown ==")
report = ev.analyze({"off": off, "lang": lang, "pool": pool}, gts)
md = ev.to_markdown(report)
check("pool" in report["reasons"] and "2nd pass samples" in report["metrics"]["off"]
      and "rolled-back samples" in report["metrics"]["pool"] and report["compare"]["n"] == len(rows),
      "report has reasons, subsets for every run, rolled-back samples for pool, comparison")
check("## (A)" in md and "## (B)" in md and "## (C)" in md, "markdown has the three parts")
try:
    ev.analyze({"off": off, "lang": lang[:-1]}, gts)
    check(False, "different sample counts must fail")
except AssertionError:
    check(True, "different sample counts -> error (runs must come from the same data)")

import pickle
import tempfile

with tempfile.TemporaryDirectory() as d:  # the same shape tools/test.py --out writes: [{"img_bbox": {...}}, ...]
    path = os.path.join(d, "run.pkl")
    with open(path, "wb") as f:
        pickle.dump([{"img_bbox": r} for r in pool], f)
    back = ev.load_results(path)
check(len(back) == N and torch.equal(back[3]["final_planning"], pool[3]["final_planning"]) and back[3]["rollback"],
      "load_results reads a tools/test.py --out pkl")

print("\nALL OK" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
