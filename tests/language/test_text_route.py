"""CPU tests for the text route (scene as a sentence, LoRA only):
  * GT info pkl and driving-model packet produce the same sentence format
  * a text-mode head learns 'Meta-action / Reason' and LanguageModule builds the text from A's packet dicts
  * train_language.py --pkl runs end to end (stage 2 only, no projector in the checkpoint)

    pytest tests/language -q
"""
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_language_pipeline import REASONS, ROOT, lang, make_builder, make_inputs, tiny

pytest.importorskip("peft")
TOOLS = ROOT / "tools" / "language"
METAS_RLS = [6, 5, 9, 11]


def make_info(i):
    deltas = np.tile(np.array([[0.3 * (i - 1.5), 1.0 + i]], dtype=np.float32), (6, 1))
    boxes = np.zeros((3, 9), dtype=np.float32)
    boxes[:, :2] = [[3.0, 10.0 + i], [-2.0, 5.0], [1.0, 1.0]]
    return dict(token=f"tok{i}", gt_ego_fut_trajs=deltas, gt_boxes=boxes,
                gt_names=np.array(["car", "pedestrian", "ignore"]), valid_flag=np.array([True, i % 2 == 0, True]))


@pytest.fixture(scope="module")
def text_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("text")
    with open(d / "infos.pkl", "wb") as f:
        pickle.dump(dict(infos=[make_info(i) for i in range(4)]), f)
    with open(d / "labels.jsonl", "w") as f:
        for i, m in enumerate(METAS_RLS):
            f.write(json.dumps(dict(sample_token=f"tok{i}", meta_action=m, reason=REASONS[i])) + "\n")
    return d


def test_scene_text_from_info_and_packet():
    text = lang.scene_text_from_info(make_info(1))
    assert text.startswith("Planned path (x right, y forward, m, every 0.5s): (-0.2,2.0)")
    assert "(-0.9,12.0)" in text  # offsets are accumulated into positions
    assert "car 11m ahead, 3m right" in text
    assert "pedestrian" not in text and "ignore" not in text  # invalid box / non-detection class dropped

    pkt = make_builder()(**make_inputs(seed=1))
    one = {k: v[0] for k, v in pkt.items()}
    ptext = lang.scene_text_from_packet(one)
    assert ptext.split(":")[0] == text.split(":")[0] and "\nNearby objects: " in ptext


def test_text_head_learns_and_module_builds_text_from_packets(text_data):
    ds = lang.ReasonTextDataset(str(text_data / "labels.jsonl"), str(text_data / "infos.pkl"))
    assert ds.stats == dict(kept=4, missing_info=0, invalid_reason=0)
    targets = [lang.format_target(m, r) for m, r in zip(METAS_RLS, REASONS)]
    lm, tok = tiny.build_tiny_lm_and_tokenizer((REASONS + targets + list(ds.texts.values()) + ["Driving scene:"]) * 4)
    torch.manual_seed(0)
    head = lang.LanguageHead(lm, tok, predict_action=True, input_mode="text")
    assert head.projector is None
    for p in head.lm.parameters():
        p.requires_grad_(True)
    batch = lang.collate_fn([ds[i] for i in range(4)])
    assert "scene" not in batch and len(batch["text"]) == 4
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        head(batch)["loss"].backward()
        opt.step()

    module = lang.LanguageModule(head=head, max_new_tokens=48)
    module.infer_batch([dict(text=t) for t in batch["text"]])
    parsed = [p.meta_idx for p in module.last_parse]
    assert sum(m == e for m, e in zip(parsed, METAS_RLS)) >= 3, [p.error for p in module.last_parse]

    seen = []
    head.generate = lambda b, **kw: seen.extend(b["text"]) or ["garbage"] * len(b["text"])
    pkt = make_builder()(**make_inputs(seed=2))
    res = module.infer_batch([{k: v[i] for k, v in pkt.items()} for i in range(2)])  # A's split_packet dicts
    assert all(t.startswith("Planned path") for t in seen)
    assert [m for m, _ in res] == [int(pkt["meta_action"][i]) for i in range(2)]  # unparsable -> rule fallback


def test_train_script_text_route(text_data, tmp_path):
    run = dict(cwd=str(ROOT), capture_output=True, text=True)
    base = [sys.executable, str(TOOLS / "train_language.py"), "--tiny", "--predict-action",
            "--labels", str(text_data / "labels.jsonl"), "--pkl", str(text_data / "infos.pkl"),
            "--max-steps", "2", "--batch-size", "2", "--workers", "0", "--log-every", "1"]
    r = subprocess.run(base + ["--stage", "1", "--out", str(tmp_path / "s1")], **run)
    assert r.returncode != 0 and "--stage 2" in r.stderr
    r = subprocess.run(base + ["--stage", "2", "--out", str(tmp_path / "ck")], **run)
    assert r.returncode == 0, r.stderr
    ck = tmp_path / "ck" / "last"
    assert json.loads((ck / "language_head.json").read_text())["input_mode"] == "text"
    assert (ck / "lora.pt").exists() and not (ck / "projector.pt").exists()
