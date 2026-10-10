"""CPU-only smoke tests for the language branch (no GPU, no mmcv, no downloads).

    pytest tests/language/test_language_pipeline.py -q

What is checked
  1. prompt tables stay in sync with models/motion/meta_action.py
  2. LangPacketBuilder: shapes, selected mode stays inside the cmd group, meta-action matches MetaActionHead
  3. ReasonPacketDataset + collate
  4. tiny LM: loss is finite, loss only on reason tokens, the model can overfit 4 samples, generate() works,
     save/load round trip gives identical generations
  5. LoRA path (skipped if peft is not installed)
"""
import json
import sys
from importlib import import_module
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "language"))
from _compat import load_language_package, load_meta_action_module  # noqa: E402

ma = load_meta_action_module(ROOT)
lang = load_language_package(ROOT)
prompt = import_module(lang.__name__ + ".prompt")
packet_mod = import_module(lang.__name__ + ".packet")
tiny = import_module(lang.__name__ + ".tiny")

B, N, D, ME, T = 3, 20, 32, 6, 6
M = 3 * ME


def make_inputs(seed=0, with_cmd=True):
    g = torch.Generator().manual_seed(seed)
    plan_reg = torch.randn(B, 1, M, T, 2, generator=g) * 0.8
    plan_reg[..., 1] += 2.0  # drive forward (y)
    cmd = torch.nn.functional.one_hot(torch.tensor([0, 1, 2]), 3).float()
    return dict(
        agent_feat=torch.randn(B, N, D, generator=g),
        agent_conf=torch.rand(B, N, generator=g),
        agent_boxes=torch.randn(B, N, 11, generator=g) * 10,
        agent_labels=torch.randint(0, 10, (B, N), generator=g),
        ego_feat_modes=torch.randn(B, M, D, generator=g),
        plan_reg=plan_reg,
        plan_cls=torch.randn(B, 1, M, generator=g),
        cmd=cmd if with_cmd else None,
        map_feat=torch.randn(B, 10, D, generator=g),
    )


def make_builder():
    head = ma.MetaActionHead(ego_fut_ts=T, ego_fut_mode=ME)
    return packet_mod.LangPacketBuilder(num_agents=8, num_map=4, ego_fut_ts=T, ego_fut_mode=ME, meta_action_head=head)


# ---------------------------------------------------------------------------- 1. prompt tables
def test_tables_match_meta_action():
    assert prompt.LATERAL_NAMES == ma.LATERAL_NAMES
    assert prompt.LONGITUDINAL_NAMES == ma.LONGITUDINAL_NAMES
    assert prompt.NUM_META_ACTIONS == ma.NUM_META_ACTIONS == 12
    for i in range(12):
        assert prompt.action_tag(i) == ma.MetaActionHead.to_text(i)
        assert len(prompt.action_phrase(i, "en")) > 5 and len(prompt.action_phrase(i, "ko")) > 2


def test_consistency_and_validation():
    left_slow = 6  # left(1)*4 + decelerate(2)
    assert prompt.action_consistency("Turn left while slowing down for the crossing pedestrian.", left_slow) == 1.0
    assert prompt.action_consistency("Go straight because the road is clear.", left_slow) == 0.0
    assert prompt.validate_reason("Slowing down because a pedestrian is crossing ahead.")
    assert not prompt.validate_reason("In the image there is a car.")
    assert not prompt.validate_reason("ok")
    assert prompt.clean_reason('Reason:  "Stop at the red light." ') == "Stop at the red light."


def test_describe_objects():
    text = prompt.describe_objects([[-3.2, 12.1], [2.0, -5.0], [0.0, 80.0]], [0, 8, 0], [0.9, 0.8, 0.9])
    assert "car 12m ahead, 3m left" in text and "pedestrian 5m behind, 2m right" in text
    assert "80" not in text  # beyond max_dist
    assert prompt.describe_objects([[0, 5]], [0], [0.1]) is None  # below min_conf


# ---------------------------------------------------------------------------- 2. packet builder
def test_packet_shapes_and_selection():
    builder = make_builder()
    inp = make_inputs()
    pkt = builder(**inp)
    L = 1 + 8 + 4
    assert pkt["scene"].shape == (B, L, D) and pkt["scene"].dtype == torch.float16
    assert pkt["scene_mask"].shape == (B, L) and pkt["scene_type"].shape == (B, L)
    assert pkt["traj"].shape == (B, T, 2)
    assert pkt["scene_type"][0].tolist() == [0] + [1] * 8 + [2] * 4
    # selected mode lies inside the commanded group
    groups = pkt["selected_mode"] // ME
    assert groups.tolist() == inp["cmd"].argmax(-1).tolist()
    # meta-action equals the rule applied to the selected trajectory
    ref = ma.MetaActionHead(ego_fut_ts=T, ego_fut_mode=ME)(inp["plan_reg"], inp["plan_cls"], inp["cmd"])
    assert torch.equal(pkt["meta_action"], ref["selected_meta_action"])
    # traj is the cumulative sum of the selected offsets
    b = torch.arange(B)
    assert torch.allclose(pkt["traj"], inp["plan_reg"].cumsum(-2)[b, 0, pkt["selected_mode"]])
    # ego token is the selected mode's feature
    assert torch.allclose(pkt["scene"][:, 0].float(), inp["ego_feat_modes"][b, pkt["selected_mode"]], atol=1e-2)
    # agents sorted by confidence, no grad
    assert (pkt["agent_conf"][:, :-1] >= pkt["agent_conf"][:, 1:]).all()
    assert not pkt["scene"].requires_grad


def test_packet_without_cmd_and_with_padding():
    builder = make_builder()
    inp = make_inputs(with_cmd=False)
    inp["map_feat"] = inp["map_feat"][:, :2]  # fewer map tokens than requested -> padded + masked
    inp["agent_feat"], inp["agent_conf"] = inp["agent_feat"][:, :5], inp["agent_conf"][:, :5]
    inp["agent_boxes"], inp["agent_labels"] = inp["agent_boxes"][:, :5], inp["agent_labels"][:, :5]
    pkt = builder(**inp)
    assert pkt["scene"].shape == (B, 13, D)
    assert pkt["scene_mask"][:, 1 + 5 : 1 + 8].sum() == 0  # padded agents masked
    assert pkt["scene_mask"][:, 1 + 8 : 1 + 8 + 2].all() and pkt["scene_mask"][:, 1 + 8 + 2 :].sum() == 0
    assert (pkt["selected_mode"] == inp["plan_cls"].reshape(B, -1).argmax(-1)).all()


# ---------------------------------------------------------------------------- 3/4. dataset + LM
REASONS = [
    "Slowing down because a pedestrian is crossing ahead of the car.",
    "Turning left at the junction since the way is clear of traffic.",
    "Going straight at a steady speed as the lane ahead is empty.",
    "Coming to a stop because the traffic light in front is red.",
]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("lang")
    builder = make_builder()
    pkt = builder(**make_inputs(seed=1))
    ids = [f"tok{i}" for i in range(B)]
    packet_mod.save_packets(pkt, ids, str(d / "packets"))
    # one more packet so that we have 4 samples
    pkt2 = builder(**make_inputs(seed=2))
    packet_mod.save_packets({k: v[:1] for k, v in pkt2.items()}, ["tok3"], str(d / "packets"))
    metas = [6, 5, 9, 11]
    with open(d / "labels.jsonl", "w") as f:
        for i in range(4):
            f.write(json.dumps(dict(sample_token=f"tok{i}", meta_action=metas[i], reason=REASONS[i])) + "\n")
        f.write(json.dumps(dict(sample_token="missing", meta_action=0, reason=REASONS[0])) + "\n")
    return d


def test_dataset_and_collate(data_dir):
    ds = lang.ReasonPacketDataset(str(data_dir / "labels.jsonl"), str(data_dir / "packets"), with_objects=True)
    assert len(ds) == 4 and ds.stats["missing_packet"] == 1
    batch = lang.collate_fn([ds[0], ds[1]])
    assert batch["scene"].shape == (2, 13, D) and batch["scene"].dtype == torch.float32
    assert batch["meta_action"].tolist() == [6, 5]
    ds_pred = lang.ReasonPacketDataset(str(data_dir / "labels.jsonl"), str(data_dir / "packets"), meta_source="pred")
    assert ds_pred[0]["meta_action"] == ds_pred[0]["pred_meta_action"]


@pytest.fixture(scope="module")
def tiny_setup(data_dir):
    corpus = REASONS + ["Planned meta-action: decelerate+left (turn left while slowing down)", "Driving scene tokens: Reason:", "Nearby objects: car"]
    lm, tok = tiny.build_tiny_lm_and_tokenizer(corpus * 4)
    ds = lang.ReasonPacketDataset(str(data_dir / "labels.jsonl"), str(data_dir / "packets"))
    return lm, tok, ds


def test_loss_is_masked_to_reason_tokens(tiny_setup):
    lm, tok, ds = tiny_setup
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=64)
    batch = lang.collate_fn([ds[i] for i in range(4)])
    emb, mask, labels = head._build(batch, with_target=True, padding_side="right")
    n_target = [len(tok(" " + r, add_special_tokens=False)["input_ids"]) + 1 for r in batch["reason"]]
    assert (labels != -100).sum(1).tolist() == n_target
    out = head(batch)
    assert torch.isfinite(out["loss"]) and out["num_target_tokens"].item() == sum(n_target)
    # only projector is trainable when no LoRA, LM stays frozen
    assert all(not p.requires_grad for p in head.lm.parameters())
    assert all(p.requires_grad for p in head.projector.parameters())


def test_frozen_lm_projector_gets_gradient_and_scene_matters(tiny_setup):
    """Stage-1 mechanics: LM frozen, gradient reaches the projector, and the scene tokens change the loss."""
    lm, tok, ds = tiny_setup
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=64)
    batch = lang.collate_fn([ds[i] for i in range(4)])
    head(batch)["loss"].backward()
    grads = [p.grad.norm().item() for p in head.projector.parameters() if p.grad is not None]
    assert grads and max(grads) > 0
    shuffled = dict(batch, scene=batch["scene"].flip(0), traj=batch["traj"].flip(0))
    with torch.no_grad():
        assert abs(head(batch)["loss"].item() - head(shuffled)["loss"].item()) > 1e-6


def test_stage1_projector_learns_to_pick_the_reason(tiny_setup):
    """Realistic stage 1: the LM already 'knows' the sentences, only the SCENE tells which one to say.

    All samples use the same meta-action, so the prompt text carries no information.
    Phase A teaches the LM the sentence distribution (scene shuffled each step -> LM cannot use it);
    phase B freezes the LM and trains only the projector, whose loss must drop clearly below phase A's plateau.
    (A frozen *random tiny* LM cannot be steered perfectly, so the bar is a 20% drop, not zero.)
    """
    lm, tok, ds = tiny_setup
    import copy

    lm = copy.deepcopy(lm)
    torch.manual_seed(0)
    batch = lang.collate_fn([ds[i] for i in range(4)])
    batch["meta_action"] = torch.full((4,), 6)
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=128)
    for p in head.lm.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for _ in range(120):  # phase A
        perm = torch.randperm(4)
        b = dict(batch, scene=batch["scene"][perm], traj=batch["traj"][perm], scene_mask=batch["scene_mask"][perm])
        opt.zero_grad()
        head(b)["loss"].backward()
        opt.step()
    plateau = head(batch)["loss"].item()
    for p in head.lm.parameters():  # phase B: freeze LM, projector only
        p.requires_grad_(False)
    opt = torch.optim.AdamW(head.trainable_parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = head(batch)["loss"]
        loss.backward()
        opt.step()
    assert loss.item() < plateau * 0.8, (plateau, loss.item())


def test_scene_information_reaches_the_output(tiny_setup):
    """Same meta-action everywhere -> only scene/traj tokens can tell the 4 samples apart.
    With a trainable LM the model must produce the right sentence per sample, proving the information path works."""
    lm, tok, ds = tiny_setup
    import copy

    lm = copy.deepcopy(lm)
    torch.manual_seed(0)
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=128)
    for p in head.lm.parameters():
        p.requires_grad_(True)
    batch = lang.collate_fn([ds[i] for i in range(4)])
    batch["meta_action"] = torch.full((4,), 6)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = head(batch)["loss"]
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    head.eval()
    gen = head.generate(batch, max_new_tokens=24)
    assert [g[:12] for g in gen] == [r[:12] for r in batch["reason"]], gen


def test_overfit_generate_and_save_load(tiny_setup, tmp_path):
    """Loss path + generate() + checkpoint round trip (LM unfrozen here so 4 sentences can be memorised)."""
    lm, tok, ds = tiny_setup
    import copy

    lm = copy.deepcopy(lm)
    torch.manual_seed(0)
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=128)
    for p in head.lm.parameters():
        p.requires_grad_(True)
    batch = lang.collate_fn([ds[i] for i in range(4)])
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    first = head(batch)["loss"].item()
    for _ in range(150):
        opt.zero_grad()
        loss = head(batch)["loss"]
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2, (first, loss.item())

    head.eval()
    gen = head.generate(batch, max_new_tokens=24)
    assert len(gen) == 4 and all(isinstance(g, str) for g in gen)
    assert sum(g[:12] == r[:12] for g, r in zip(gen, batch["reason"])) >= 3, gen

    head.save(str(tmp_path / "ckpt"))
    head2 = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=128).load(str(tmp_path / "ckpt")).eval()
    assert head2.generate(batch, max_new_tokens=24) == gen  # same LM object, projector restored from disk


def test_lora_path(tiny_setup, tmp_path):
    pytest.importorskip("peft")
    _, _, ds = tiny_setup
    corpus = REASONS * 4
    lm, tok = tiny.build_tiny_lm_and_tokenizer(corpus)
    head = lang.LanguageHead(lm, tok, scene_dim=D, traj_steps=T, projector_hidden=64, lora_cfg=dict(r=4, lora_alpha=8))
    batch = lang.collate_fn([ds[i] for i in range(2)])
    n_lora = sum(p.numel() for n, p in head.lm.named_parameters() if p.requires_grad)
    assert n_lora > 0
    head(batch)["loss"].backward()
    assert any(p.grad is not None for n, p in head.lm.named_parameters() if "lora_" in n and p.requires_grad)
    head.save(str(tmp_path / "c"))
    assert (tmp_path / "c" / "lora.pt").exists()
    lm2, _ = tiny.build_tiny_lm_and_tokenizer(corpus)
    head2 = lang.LanguageHead(lm2, tok, scene_dim=D, traj_steps=T, projector_hidden=64, lora_cfg=dict(r=4, lora_alpha=8)).load(str(tmp_path / "c"))
    for (n1, p1), (n2, p2) in zip(head.lm.named_parameters(), head2.lm.named_parameters()):
        if "lora_" in n1:
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------- 5. packet dumper (forward hook)
def test_packet_dumper_hook(tmp_path):
    builder = make_builder()

    class MotionPlanningHead(torch.nn.Module):  # same class name the dumper looks for
        def __init__(self):
            super().__init__()
            self.lang_packet_builder = builder

        def forward(self, det_output, map_output, feature_maps, metas, *rest):
            inp = make_inputs(seed=5)
            return {}, {"lang_packet": builder(**inp)}

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = MotionPlanningHead()

        def forward(self, metas):
            return self.head(None, None, None, metas)

    model = Wrapper()
    dumper = packet_mod.attach_packet_dumper(model, str(tmp_path / "pk"))
    model({"img_metas": [{"sample_idx": "a"}, {"sample_idx": "b"}, {"sample_idx": "c"}]})
    assert sorted(p.name for p in (tmp_path / "pk").iterdir()) == ["a.pt", "b.pt", "c.pt"]
    loaded = packet_mod.load_packet(str(tmp_path / "pk" / "b.pt"))
    assert loaded["scene"].shape == (1 + 8 + 4, D) and loaded["meta_action"].dim() == 0
    model({"img_metas": [{"sample_idx": "a"}, {"sample_idx": "b"}, {"sample_idx": "c"}]})  # no overwrite by default
    assert dumper.num_saved == 3
    dumper.remove()
    model({"img_metas": [{"sample_idx": "x"}, {"sample_idx": "y"}, {"sample_idx": "z"}]})  # removed -> nothing saved
    assert not (tmp_path / "pk" / "x.pt").exists()
    # metas passed as a keyword argument
    kw = packet_mod.attach_packet_dumper(model, str(tmp_path / "pk_kw"))
    model.head(None, None, None, metas={"img_metas": [{"sample_idx": s} for s in "def"]})
    assert sorted(p.name for p in (tmp_path / "pk_kw").iterdir()) == ["d.pt", "e.pt", "f.pt"]
    kw.remove()
    # fallback ids when metas carry no token
    model2 = Wrapper()
    packet_mod.attach_packet_dumper(model2, str(tmp_path / "pk2"))
    model2({})
    assert sorted(p.name for p in (tmp_path / "pk2").iterdir()) == ["idx0000000.pt", "idx0000001.pt", "idx0000002.pt"]
