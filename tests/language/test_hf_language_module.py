"""CPU tests for the real language module path (tiny random Qwen2, no downloads):
  * LanguageHead --predict-action learns to write 'Meta-action: ...\\nReason: ...' and HFLanguageModule parses it
  * mock-VLM label generation -> train_language.py --tiny --predict-action runs end to end

    pytest tests/language -q
"""
import json
import pickle
import subprocess
import sys
from importlib import import_module

import pytest
import torch

from test_language_pipeline import REASONS, ROOT, B, D, T, lang, make_builder, make_inputs, packet_mod, tiny

parser = import_module(lang.__name__ + ".output_parser")
lm_mod = import_module(lang.__name__ + ".language_module")
TOOLS = ROOT / "tools" / "language"
METAS_RLS = [6, 5, 9, 11]  # label files use meta_action.py numbering


@pytest.fixture(scope="module")
def action_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("lm")
    builder = make_builder()
    packet_mod.save_packets(builder(**make_inputs(seed=1)), [f"tok{i}" for i in range(B)], str(d / "packets"))
    pkt2 = builder(**make_inputs(seed=2))
    packet_mod.save_packets({k: v[:1] for k, v in pkt2.items()}, ["tok3"], str(d / "packets"))
    with open(d / "labels.jsonl", "w") as f:
        for i, m in enumerate(METAS_RLS):
            f.write(json.dumps(dict(sample_token=f"tok{i}", meta_action=m, reason=REASONS[i])) + "\n")
    return d


def test_hf_module_predicts_meta_and_reason(action_data):
    ds = lang.ReasonPacketDataset(str(action_data / "labels.jsonl"), str(action_data / "packets"))
    targets = [parser.format_target(lang.from_rls_index(m), r) for m, r in zip(METAS_RLS, REASONS)]
    lm, tok = tiny.build_tiny_lm_and_tokenizer((REASONS + targets + ["Driving scene tokens:"]) * 4)
    torch.manual_seed(0)
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=128, predict_action=True)
    for p in head.lm.parameters():
        p.requires_grad_(True)
    batch = lang.collate_fn([ds[i] for i in range(4)])
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        head(batch)["loss"].backward()
        opt.step()

    module = lm_mod.HFLanguageModule(head=head, max_new_tokens=48)
    inputs = [ds[i] for i in range(4)]  # per-sample dicts, no batch dim (interface contract)
    meta, reason = module.infer(inputs[0])
    assert meta is None or isinstance(meta, int)
    results = module.infer_batch(inputs)
    expected = [lang.from_rls_index(m) for m in METAS_RLS]
    parsed = [p.meta_idx for p in module.last_parse]  # before the rule fallback
    assert sum(m == e for m, e in zip(parsed, expected)) >= 3, (results, [p.error for p in module.last_parse])
    assert sum((r[1] or "")[:12] == g[:12] for r, g in zip(results, REASONS)) >= 3, results


def test_hf_module_rejects_reason_only_head():
    lm, tok = tiny.build_tiny_lm_and_tokenizer(REASONS)
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=16)
    with pytest.raises(AssertionError):
        lm_mod.HFLanguageModule(head=head)


def test_mock_label_pipeline_then_train(action_data, tmp_path):
    infos = [dict(token=f"tok{i}", cams=dict(CAM_FRONT=dict(data_path=f"img{i}.jpg"))) for i in range(4)]
    with open(tmp_path / "infos.pkl", "wb") as f:
        pickle.dump(dict(infos=infos), f)
    (tmp_path / "meta.json").write_text(json.dumps({f"tok{i}": m for i, m in enumerate(METAS_RLS)}))
    labels = tmp_path / "reasons.jsonl"
    run = dict(cwd=str(ROOT), capture_output=True, text=True)

    r = subprocess.run([sys.executable, str(TOOLS / "gen_reason_labels.py"), "--mock", "--pkl", str(tmp_path / "infos.pkl"),
                        "--meta-json", str(tmp_path / "meta.json"), "--out", str(labels), "--batch-size", "3"], **run)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert [row["meta_action"] for row in rows] == METAS_RLS
    assert all(lang.validate_reason(row["reason"]) for row in rows)

    r = subprocess.run([sys.executable, str(TOOLS / "train_language.py"), "--stage", "1", "--tiny", "--predict-action",
                        "--labels", str(labels), "--packets", str(action_data / "packets"), "--out", str(tmp_path / "ck"),
                        "--scene-dim", str(D), "--max-steps", "2", "--batch-size", "2", "--workers", "0", "--log-every", "1"], **run)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "ck" / "last" / "language_head.json").read_text())["predict_action"] is True


def test_parse_failure_falls_back_to_rule_meta():
    lm, tok = tiny.build_tiny_lm_and_tokenizer(REASONS)
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=16, predict_action=True)
    head.generate = lambda batch, **kw: ["I am not sure.", "Meta-action: right+stop\nReason: red light."]
    module = lang.LanguageModule(head=head)
    x = dict(scene=torch.zeros(5, D), scene_mask=torch.ones(5, dtype=torch.bool), scene_type=torch.zeros(5, dtype=torch.long),
             traj=torch.zeros(T, 2))
    res = module.infer_batch([dict(x, meta_action=6), dict(x, meta_action=6)])
    assert res[0] == (lang.from_rls_index(6), None) and module.num_fallback == 1  # rule, meta_action.py numbering
    assert res[1] == (11, "red light.")
    head.generate = lambda batch, **kw: ["garbage"]
    assert module.infer(x) == (None, None)  # no rule meta-action in the input -> caller decides


def test_from_pretrained_8bit_builds_quant_config(monkeypatch):
    """Team setup (Qwen1.5-1.8B-Chat, 8bit + LoRA): the 8bit config reaches transformers. No GPU / download needed."""
    import transformers

    seen = {}
    lm, tok = tiny.build_tiny_lm_and_tokenizer(REASONS)

    def fake_lm(path, **kw):
        seen.update(kw)
        return lm

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_lm)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kw: tok)
    head = lang.LanguageHead.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", load_in_8bit=True, scene_dim=D, traj_steps=T)
    assert seen["quantization_config"].load_in_8bit and not seen["quantization_config"].load_in_4bit
    assert seen["device_map"] == {"": 0}
    assert head.lm is lm
    with pytest.raises(ValueError):
        lang.LanguageHead.from_pretrained("x", load_in_4bit=True, load_in_8bit=True)
