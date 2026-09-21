"""Equivalence tests for Sparse Query pruning (CPU only, no mmcv / mmdet / CUDA ops).

    python tests/test_sparse_query.py          # or: pytest tests/test_sparse_query.py

The REAL repo files are loaded:
    models/motion/motion_planning_head.py          (parent, unmodified)
    models/motion/sparse_query_motion_planning_head.py
    models/motion/sparse_query_utils.py
    models/motion/instance_queue.py                (Memory Bank)
    models/motion/motion_blocks.py                 (Multi-mode Planning refine)
    models/motion/target.py                        (MotionTarget)
    models/diffusion/distributions.py              (Generative Decoder modules)
    models/losses/generation_loss.py               (ProbabilisticLoss, Eq. 6)
mmcv / mmdet registries, attention and FFN are replaced by small stand-ins with the
same call signatures. Attention stays per-query, like the real kernels.
"""
import math
import pathlib
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "projects" / "mmdet3d_plugin"


# ═════════════════════════════════════════════════════════════════════════════
# stand-ins for mmcv / mmdet and CUDA-only modules
# ═════════════════════════════════════════════════════════════════════════════
class Registry(dict):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def register_module(self, name=None, force=False, module=None):
        def deco(cls):
            self[name or cls.__name__] = cls
            return cls

        return deco(module) if module is not None else deco


def build_from_cfg(cfg, registry, default_args=None):
    cfg = dict(cfg)
    t = cfg.pop("type")
    return (registry[t] if isinstance(t, str) else t)(**cfg)


def _passthrough_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _module(name, pkg_path=None, **attrs):
    m = types.ModuleType(name)
    if pkg_path is not None:
        m.__path__ = [str(pkg_path)]
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


ATTENTION, PLUGIN_LAYERS, POS_ENC, FFN_REG, NORM_LAYERS = (Registry(n) for n in "a p pe f n".split())
HEADS, LOSSES, BBOX_SAMPLERS, BBOX_CODERS = (Registry(n) for n in "h l s c".split())


def build_loss(cfg):
    return None if cfg is None else build_from_cfg(cfg, LOSSES)


class MHA(nn.Module):
    """mmcv MultiheadAttention semantics: identity residual, value defaults to key."""

    def __init__(self, embed_dims, num_heads, batch_first=True, dropout=0.0, **kw):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None, key_padding_mask=None, **kw):
        key = query if key is None else key
        value = key if value is None else value
        q = query if query_pos is None else query + query_pos
        k = key if key_pos is None else key + key_pos
        out, _ = self.attn(q, k, value, key_padding_mask=key_padding_mask, need_weights=False)
        return query + out


class AsymmetricFFN(nn.Module):
    def __init__(self, in_channels, embed_dims, feedforward_channels, pre_norm=None, **kw):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_channels) if pre_norm else nn.Identity()
        self.fc1 = nn.Linear(in_channels, feedforward_channels)
        self.fc2 = nn.Linear(feedforward_channels, embed_dims)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(self.pre_norm(x))))


class SimpleLoss(nn.Module):
    def __init__(self, loss_weight=1.0, **kw):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target=None, weight=None, avg_factor=None, **kw):
        return pred.float().abs().mean() * self.loss_weight


ATTENTION["MultiheadAttention"] = MHA
ATTENTION["MultiheadFlashAttention"] = MHA
FFN_REG["AsymmetricFFN"] = AsymmetricFFN
NORM_LAYERS["LN"] = lambda normalized_shape: nn.LayerNorm(normalized_shape)
for _n in ["FocalLoss", "L1Loss", "KLLoss", "MSELoss"]:
    LOSSES[_n] = SimpleLoss


def _gen_sineembed_for_position(pos_tensor, hidden_dim=256):  # verbatim from models/attention.py
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


def _linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):  # verbatim from models/blocks.py
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def _topk(confidence, k, *inputs):  # verbatim from models/instance_bank.py
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


def _install_stubs():
    _module("mmcv")
    _module("mmcv.utils", build_from_cfg=build_from_cfg, deprecated_api_warning=_passthrough_decorator)
    _module(
        "mmcv.cnn",
        Linear=nn.Linear,
        Scale=nn.Identity,
        bias_init_with_prob=lambda p: float(-math.log((1 - p) / p)),
        xavier_init=lambda *a, **k: None,
    )
    _module("mmcv.cnn.bricks")
    _module(
        "mmcv.cnn.bricks.registry",
        ATTENTION=ATTENTION,
        PLUGIN_LAYERS=PLUGIN_LAYERS,
        POSITIONAL_ENCODING=POS_ENC,
        FEEDFORWARD_NETWORK=FFN_REG,
        NORM_LAYERS=NORM_LAYERS,
    )
    _module("mmcv.runner", BaseModule=BaseModule, force_fp32=_passthrough_decorator, auto_fp16=_passthrough_decorator)
    _module("mmcv.runner.base_module", BaseModule=BaseModule, Sequential=nn.Sequential)
    _module("mmdet")
    _module("mmdet.core", reduce_mean=lambda x: x)
    _module("mmdet.core.bbox")
    _module("mmdet.core.bbox.builder", BBOX_SAMPLERS=BBOX_SAMPLERS, BBOX_CODERS=BBOX_CODERS)
    _module("mmdet.models", HEADS=HEADS, LOSSES=LOSSES, build_loss=build_loss)

    # package skeleton WITHOUT running the heavy __init__.py files
    _module("projects", ROOT / "projects")
    _module("projects.mmdet3d_plugin", PLUGIN)
    _module("projects.mmdet3d_plugin.core", PLUGIN / "core")
    _module("projects.mmdet3d_plugin.datasets", PLUGIN / "datasets")
    _module("projects.mmdet3d_plugin.datasets.utils", box3d_to_corners=lambda *a, **k: None)
    _module("projects.mmdet3d_plugin.ops", feature_maps_format=lambda *a, **k: None)
    _module("projects.mmdet3d_plugin.models", PLUGIN / "models")
    _module("projects.mmdet3d_plugin.models.motion", PLUGIN / "models" / "motion")
    _module("projects.mmdet3d_plugin.models.losses", PLUGIN / "models" / "losses")
    _module("projects.mmdet3d_plugin.models.attention", gen_sineembed_for_position=_gen_sineembed_for_position)
    _module("projects.mmdet3d_plugin.models.blocks", linear_relu_ln=_linear_relu_ln)
    _module("projects.mmdet3d_plugin.models.instance_bank", topk=_topk)


_install_stubs()
import importlib  # noqa: E402

importlib.import_module("projects.mmdet3d_plugin.models.motion.motion_blocks")
importlib.import_module("projects.mmdet3d_plugin.models.motion.instance_queue")
importlib.import_module("projects.mmdet3d_plugin.models.motion.target")
importlib.import_module("projects.mmdet3d_plugin.models.losses.generation_loss")
MPH = importlib.import_module("projects.mmdet3d_plugin.models.motion.motion_planning_head").MotionPlanningHead
SQ = importlib.import_module("projects.mmdet3d_plugin.models.motion.sparse_query_motion_planning_head")
U = importlib.import_module("projects.mmdet3d_plugin.models.motion.sparse_query_utils")
SQHead = SQ.SparseQueryMotionPlanningHead


# ═════════════════════════════════════════════════════════════════════════════
# fixtures
# ═════════════════════════════════════════════════════════════════════════════
D = 256          # sine embeddings are hard-coded to 256 in the parent forward
N_A, N_M = 300, 100   # N_A reduced from 900 only to keep the Memory Bank's [N, N, D] match small
N_E = 18
NUM_CLS = 10
FUT_TS, FUT_MODE, EGO_TS, EGO_MODE = 12, 6, 6, 6
_TMP = pathlib.Path(tempfile.mkdtemp())
np.save(_TMP / "motion.npy", np.random.RandomState(0).randn(NUM_CLS, FUT_MODE, FUT_TS, 2).astype(np.float32))
np.save(_TMP / "plan.npy", np.random.RandomState(1).randn(3, EGO_MODE, EGO_TS, 2).astype(np.float32))


def head_cfg():
    att = lambda t: dict(type=t, embed_dims=D, num_heads=8, batch_first=True, dropout=0.0)  # noqa: E731
    return dict(
        fut_ts=FUT_TS, fut_mode=FUT_MODE, ego_fut_ts=EGO_TS, ego_fut_mode=EGO_MODE,
        motion_anchor=str(_TMP / "motion.npy"), plan_anchor=str(_TMP / "plan.npy"),
        embed_dims=D, decouple_attn=False,
        instance_queue=dict(type="InstanceQueue", embed_dims=D, queue_length=4,
                            tracking_threshold=0.2, feature_map_scale=(8, 22)),
        operation_order=["temp_gnn", "gnn", "norm", "cross_gnn", "norm", "ffn", "norm"] * 3
        + ["distribution", "refine"],
        temp_graph_model=att("MultiheadAttention"),
        graph_model=att("MultiheadFlashAttention"),
        cross_graph_model=att("MultiheadFlashAttention"),
        norm_layer=dict(type="LN", normalized_shape=D),
        ffn=dict(type="AsymmetricFFN", in_channels=D, pre_norm=dict(type="LN"), embed_dims=D,
                 feedforward_channels=2 * D),
        refine_layer=dict(type="MotionPlanningRefinementModule", embed_dims=D, fut_ts=FUT_TS,
                          fut_mode=FUT_MODE, ego_fut_ts=EGO_TS, ego_fut_mode=EGO_MODE),
        motion_sampler=dict(type="MotionTarget"),
        planning_sampler=dict(type="PlanningTarget", ego_fut_ts=EGO_TS, ego_fut_mode=EGO_MODE),
        motion_loss_cls=dict(type="FocalLoss"), motion_loss_reg=dict(type="L1Loss"),
        plan_loss_cls=dict(type="FocalLoss"), plan_loss_reg=dict(type="L1Loss"),
        plan_loss_status=dict(type="L1Loss"),
        distribution_cfg=AttrDict(present_distribution_in_channels=D, future_distribution_in_channels=EGO_TS * 2,
                                  latent_dim=32, min_log_sigma=-5.0, max_log_sigma=5.0, layer_dim=4,
                                  with_cur=True, future_concate=False, log_paramter=1.0, temporal_frames=1),
        multi_modal_cfg=AttrDict(plan_instance_dim=128),
        loss_vae_gen=dict(type="ProbabilisticLoss", loss_weight=1.0, only_valid_agent=True),
        num_det=50, num_map=10,
    )


def build_pair(**sq_kwargs):
    torch.manual_seed(0)
    parent = MPH(**head_cfg())
    parent.init_weights()
    sq = SQHead(**head_cfg(), **sq_kwargs)
    sq.load_state_dict(parent.state_dict())
    return parent, sq


class AnchorHandler:
    def anchor_projection(self, anchor, T_list, time_intervals=None):
        return [anchor]


def make_frame(B, seed, n_confident=20):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, N_A, NUM_CLS, generator=g) - 4.0
    for b in range(B):
        hot = torch.randperm(N_A, generator=g)[:n_confident]
        logits[b, hot, torch.randint(0, NUM_CLS, (n_confident,), generator=g)] = 2.0
    pred = torch.zeros(B, N_A, 11)
    pred[..., :2] = torch.rand(B, N_A, 2, generator=g) * 100 - 50
    pred[..., 3:6] = torch.randn(B, N_A, 3, generator=g) * 0.3
    yaw = torch.rand(B, N_A, generator=g) * 6.28
    pred[..., 6], pred[..., 7] = yaw.sin(), yaw.cos()
    pred[..., 8:10] = torch.randn(B, N_A, 2, generator=g)
    det = {
        "instance_feature": torch.randn(B, N_A, D, generator=g),
        "anchor_embed": torch.randn(B, N_A, D, generator=g),
        "classification": [logits],
        "prediction": [pred],
        "instance_id": torch.arange(N_A).unsqueeze(0).repeat(B, 1),  # stable ids -> Memory Bank carries over
    }
    mp = {
        "instance_feature": torch.randn(B, N_M, D, generator=g),
        "anchor_embed": torch.randn(B, N_M, D, generator=g),
        "classification": [torch.randn(B, N_M, 3, generator=g)],
        "prediction": [torch.randn(B, N_M, 40, generator=g)],
    }
    metas = {"img_metas": [{"T_global": np.eye(4), "T_global_inv": np.eye(4)} for _ in range(B)]}
    return det, mp, metas


def run(head, frame, seed, anchor_encoder, B):
    det, mp, metas = frame
    torch.manual_seed(seed)  # same RNG state before the Generative Decoder noise draw
    return head(det, mp, [torch.zeros(1)], metas, anchor_encoder, torch.ones(B, dtype=torch.bool), AnchorHandler())


def close(a, b, atol=2e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-5)


def maxdiff(a, b):
    return float((a - b).abs().max())


# ═════════════════════════════════════════════════════════════════════════════
# 1. pure utilities
# ═════════════════════════════════════════════════════════════════════════════
def test_gather_scatter_roundtrip():
    x = torch.randn(2, 10, 3, 4)
    idx = torch.tensor([[1, 4, 7], [0, 2, 9]])
    y = U.gather_tokens(x, idx)
    z = U.scatter_tokens(y, idx, 10, -60.0)
    assert close(U.gather_tokens(z, idx), y)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, idx[0]] = False
    mask[1, idx[1]] = False
    assert (z[mask] == -60.0).all()


def test_gru_source_mask_matches_original_reshape():
    """The analytic source formula must agree with tracing the real reshape."""
    for B, N, E, L in [(1, 900, 18, 4), (2, 900, 18, 4), (3, 300, 18, 4)]:
        N1 = N + E
        token_id = torch.arange(B * N1, dtype=torch.float64).view(B, N1, 1).expand(B, N1, 256).contiguous()
        h = U.gru_hidden_original(token_id, L)  # [L, B*N1, 64]
        rows = torch.zeros(B, N1, dtype=torch.bool)
        rows[:, N:] = True
        traced = torch.zeros(B, N, dtype=torch.bool)
        for r in rows.reshape(-1).nonzero().flatten():
            for l in range(L):
                t = int(h[l, r, 0])
                if t % N1 < N:
                    traced[t // N1, t % N1] = True
        assert torch.equal(U.gru_source_mask(rows, N, L), traced), (B, N)
    # the headline fact for the released model (B=1, N_A=900)
    src = U.gru_source_mask(torch.cat([torch.zeros(1, 900, dtype=torch.bool), torch.ones(1, 18, dtype=torch.bool)], 1), 900, 4)
    assert src[0].nonzero().flatten().tolist() == list(range(225, 230)) + list(range(454, 459)) + list(range(684, 689))


def test_select_queries_keeps_must_and_sorts():
    conf = torch.rand(2, 50)
    must = torch.zeros(2, 50, dtype=torch.bool)
    must[0, [3, 5]] = True
    must[1, list(range(12))] = True  # more than K -> K_eff grows
    idx, k = U.select_queries(conf, 8, must)
    assert k == 12 and idx.shape == (2, 12)
    assert torch.equal(idx, idx.sort(dim=1).values)
    for b in range(2):
        assert set(must[b].nonzero().flatten().tolist()) <= set(idx[b].tolist())


def test_remap_match_indices():
    keep = torch.tensor([[2, 5, 9]])
    inv = U.build_inverse_index(keep, 12)
    kept, mapped = U.remap_match_indices([[torch.tensor([5, 7, 9]), torch.tensor([0, 1, 2])]], inv)
    assert kept[0][0].tolist() == [5, 9] and kept[0][1].tolist() == [0, 2]
    assert mapped[0][0].tolist() == [1, -1, 2] and mapped[0][1].tolist() == [0, 1, 2]
    kept, mapped = U.remap_match_indices([[None, None]], inv)
    assert kept[0] == (None, None)


def test_probabilistic_loss_matches_repo_when_nothing_pruned():
    PL = LOSSES["ProbabilisticLoss"](loss_weight=1.0, only_valid_agent=True, ego_mode_num=N_E)
    torch.manual_seed(3)
    B, K, C = 2, 40, 32
    out = {k: torch.randn(B, K + N_E, C) * 0.3 for k in ["present_mu", "present_log_sigma", "future_mu", "future_log_sigma"]}
    indices = []
    gt_idx = []
    for b in range(B):
        n = 12
        indices.append((torch.randperm(K)[:n], torch.arange(n)))
        gt_idx.append(torch.tensor(sorted(torch.randperm(n)[:8].tolist())))
    ref = PL({k: v.clone() for k, v in out.items()}, {"indices": indices}, gt_idx)["distribution_loss"]
    mine = U.probabilistic_loss_pruned(out, indices, gt_idx, N_E, 1.0)
    assert close(ref, mine, atol=1e-6), (float(ref), float(mine))


def test_probabilistic_loss_skips_pruned_without_shifting_slots():
    torch.manual_seed(4)
    B, K, C = 1, 20, 8
    out = {k: torch.randn(B, K + N_E, C) * 0.3 for k in ["present_mu", "present_log_sigma", "future_mu", "future_log_sigma"]}
    p = torch.tensor([4, -1, 7])      # GT #1 matched a pruned instance
    t = torch.tensor([0, 1, 2])
    gt_idx = [torch.tensor([0, 1, 2])]
    got = U.probabilistic_loss_pruned(out, [(p, t)], gt_idx, N_E, 1.0)
    # expected: future slots 0 and 2 (NOT 0 and 1) paired with instances 4 and 7, plus ego
    pm, pls, fm, fls = (out[k][0] for k in ["present_mu", "present_log_sigma", "future_mu", "future_log_sigma"])
    ego = list(range(K, K + N_E))
    inst, slot = [4, 7] + ego, [0, 2] + ego
    kl = pls[inst] - fls[slot] - 0.5 + (torch.exp(2 * fls[slot]) + (fm[slot] - pm[inst]) ** 2) / (2 * torch.exp(2 * pls[inst]))
    assert close(got, kl.sum(-1).mean(), atol=1e-6)


def test_motion_target_on_scattered_outputs():
    """MotionTarget (real) sees N_A slots after scatter-back; pruned matches are dropped."""
    MT = BBOX_SAMPLERS["MotionTarget"]()
    keep = torch.tensor([[1, 3, 6]])
    reg_k = torch.randn(1, 3, FUT_MODE, FUT_TS, 2)
    reg_full = U.scatter_tokens(reg_k, keep, 8, 0.0)
    gt = [torch.randn(4, FUT_TS, 2)]
    gt_mask = [torch.ones(4, FUT_TS)]
    inv = U.build_inverse_index(keep, 8)
    kept, _ = U.remap_match_indices([[torch.tensor([1, 2, 6, 7]), torch.tensor([0, 1, 2, 3])]], inv)
    _, cls_w, _, reg_t, reg_w, num_pos = MT.sample(reg_full, gt, gt_mask, {"indices": kept})
    assert int(num_pos) == 2
    assert cls_w[0].nonzero().flatten().tolist() == [1, 6]
    assert close(reg_t[0, 6], gt[0][2]) and (reg_w[0, [0, 2, 3, 4, 5, 7]] == 0).all()


# ═════════════════════════════════════════════════════════════════════════════
# 2. full forward vs the unmodified parent (2 frames, Memory Bank carried over)
# ═════════════════════════════════════════════════════════════════════════════
def _two_frame_compare(B, **sq_kwargs):
    parent, sq = build_pair(**sq_kwargs)
    parent.eval()
    sq.eval()
    torch.manual_seed(5)
    enc = nn.Linear(11, D)
    results = []
    with torch.no_grad():
        for f in range(2):
            frame = make_frame(B, seed=10 + f)
            mo_p, po_p = run(parent, frame, 100 + f, enc, B)
            mo_s, po_s = run(sq, frame, 100 + f, enc, B)
            results.append((frame, mo_p, po_p, mo_s, po_s, dict(sq.sq_stats), sq.keep_idx))
    return results


def _assert_planning_equal(po_p, po_s):
    for key in ["classification", "prediction", "status", "feature"]:
        assert close(po_p[key][-1], po_s[key][-1]), (key, maxdiff(po_p[key][-1], po_s[key][-1]))
    for a, b in zip(po_p["decoder_feature"], po_s["decoder_feature"]):
        assert close(a, b), ("decoder_feature", maxdiff(a, b))
    assert close(po_p["encoder_feature"], po_s["encoder_feature"])


def test_preserve_mode_planning_is_identical_B1():
    for frame, mo_p, po_p, mo_s, po_s, stats, keep in _two_frame_compare(1, num_query=64):
        assert keep is not None and keep.shape[1] == stats["k_eff"] < N_A
        _assert_planning_equal(po_p, po_s)
        # rescore agents (conf >= 0.5): motion identical -> final_planning rescore identical
        conf = frame[0]["classification"][-1].sigmoid().max(-1).values
        rs = conf[0] >= 0.5
        assert close(mo_p["prediction"][-1][0, rs], mo_s["prediction"][-1][0, rs])
        assert close(mo_p["classification"][-1][0, rs], mo_s["classification"][-1][0, rs])
        # pruned slots carry the fill values
        pruned = torch.ones(N_A, dtype=torch.bool)
        pruned[keep[0]] = False
        assert (mo_s["classification"][-1][0, pruned] == -60.0).all()
        assert (mo_s["prediction"][-1][0, pruned] == 0).all()
        assert mo_s["prediction"][-1].shape == mo_p["prediction"][-1].shape


def test_preserve_mode_planning_is_identical_B2_cross_sample():
    """B>1: GRU h0 sources cross into the other sample; preserve mode must follow them."""
    for frame, mo_p, po_p, mo_s, po_s, stats, keep in _two_frame_compare(2, num_query=64):
        _assert_planning_equal(po_p, po_s)


def test_original_mode_changes_planning():
    """Naive pruning (original reshape on the shorter tensor) moves ego's GRU h0 sources."""
    diffs = [maxdiff(r[2]["prediction"][-1], r[4]["prediction"][-1])
             for r in _two_frame_compare(1, num_query=64, generative_hidden="original")]
    assert max(diffs) > 1e-4, diffs


def _zero_noise(head):
    """Make the Generative Decoder sample = mu so runs with different K draw no noise."""
    def dist_fwd(present_features, future_distribution_inputs=None, noise=None):
        z = torch.zeros(present_features.shape[0], present_features.shape[1], head.latent_dim)
        return MPH.distribution_forward(head, present_features, future_distribution_inputs, noise=z)
    head.distribution_forward = dist_fwd


def test_per_token_mode_is_independent_of_K():
    torch.manual_seed(0)
    a = SQHead(**head_cfg(), num_query=64, generative_hidden="per_token")
    a.init_weights()
    b = SQHead(**head_cfg(), num_query=150, generative_hidden="per_token")
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()
    _zero_noise(a)
    _zero_noise(b)
    enc = nn.Linear(11, D)
    frame = make_frame(1, seed=21)
    with torch.no_grad():
        mo_a, po_a = run(a, frame, 7, enc, 1)
        mo_b, po_b = run(b, frame, 7, enc, 1)
    _assert_planning_equal(po_a, po_b)
    common = sorted(set(a.keep_idx[0].tolist()) & set(b.keep_idx[0].tolist()))
    assert len(common) >= 64
    assert close(mo_a["prediction"][-1][0, common], mo_b["prediction"][-1][0, common])


def test_num_query_none_is_parent():
    parent, sq = build_pair(num_query=None)
    parent.eval()
    sq.eval()
    enc = nn.Linear(11, D)
    frame = make_frame(1, seed=31)
    with torch.no_grad():
        mo_p, po_p = run(parent, frame, 9, enc, 1)
        mo_s, po_s = run(sq, frame, 9, enc, 1)
    assert sq.keep_idx is None
    _assert_planning_equal(po_p, po_s)
    assert close(mo_p["prediction"][-1], mo_s["prediction"][-1])


# ═════════════════════════════════════════════════════════════════════════════
# 3. training path: forward + loss() run end to end
# ═════════════════════════════════════════════════════════════════════════════
def test_training_forward_and_loss():
    torch.manual_seed(0)
    sq = SQHead(**head_cfg(), num_query=64)
    sq.init_weights()
    sq.train()
    B = 2
    det, mp, metas = make_frame(B, seed=41)
    anchors_xy = det["prediction"][-1][..., :2]
    gt_boxes, labels, trajs, masks, indices = [], [], [], [], []
    for b in range(B):
        n = 90 if b == 0 else 30                      # sample 0 has more vehicles than K
        slots = torch.randperm(N_A)[:n]
        box = torch.zeros(n, 9)
        box[:, :2] = anchors_xy[b, slots]              # GT centred on those anchors
        gt_boxes.append(box)
        labels.append(torch.zeros(n, dtype=torch.long))  # all 'car' -> all vehicles
        trajs.append(torch.randn(n, FUT_TS, 2))
        masks.append(torch.ones(n, FUT_TS))
        order = torch.argsort(slots)
        indices.append([slots[order], torch.arange(n)[order]])
    metas.update(
        gt_bboxes_3d=gt_boxes, gt_labels_3d=labels, gt_agent_fut_trajs=trajs, gt_agent_fut_masks=masks,
        gt_ego_fut_trajs=torch.randn(B, EGO_TS, 2), gt_ego_fut_masks=torch.ones(B, EGO_TS),
        gt_ego_fut_cmd=torch.eye(3)[torch.tensor([0, 2])], ego_status=torch.randn(B, 10),
    )
    enc = nn.Linear(11, D)
    mo, po = sq(det, mp, [torch.zeros(1)], metas, enc, torch.ones(B, dtype=torch.bool), AnchorHandler())
    k = sq.keep_idx.shape[1]
    assert k >= 90, "force_positive must keep every GT-centred anchor"
    assert mo["prediction"][-1].shape == (B, N_A, FUT_MODE, FUT_TS, 2)
    assert po["distribution"]["present_mu"].shape[1] == k + N_E
    assert po["distribution"]["future_mu"].shape[1] == k + N_E
    for b in range(B):
        assert set(indices[b][0].tolist()) <= set(sq.keep_idx[b].tolist())
    losses = sq.loss(mo, po, metas, {"indices": indices})
    assert "distribution_loss" in losses and torch.isfinite(losses["distribution_loss"])
    total = sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
    total.backward()


def test_training_without_force_positive_truncates_and_skips():
    """More vehicles than K and pruned positives: no crash, no slot misalignment."""
    torch.manual_seed(0)
    sq = SQHead(**head_cfg(), num_query=64, force_positive=False)
    sq.init_weights()
    sq.train()
    B = 1
    det, mp, metas = make_frame(B, seed=43)
    n = 120
    slots = torch.randperm(N_A)[:n]
    order = torch.argsort(slots)
    indices = [[slots[order], torch.arange(n)[order]]]
    metas.update(
        gt_labels_3d=[torch.zeros(n, dtype=torch.long)], gt_agent_fut_trajs=[torch.randn(n, FUT_TS, 2)],
        gt_agent_fut_masks=[torch.ones(n, FUT_TS)], gt_ego_fut_trajs=torch.randn(B, EGO_TS, 2),
        gt_ego_fut_masks=torch.ones(B, EGO_TS), gt_ego_fut_cmd=torch.eye(3)[[1]], ego_status=torch.randn(B, 10),
    )
    enc = nn.Linear(11, D)
    mo, po = sq(det, mp, [torch.zeros(1)], metas, enc, torch.ones(B, dtype=torch.bool), AnchorHandler())
    k = sq.keep_idx.shape[1]
    assert k < n and sq.sq_stats["n_future_truncated"] == n - k
    assert len(sq.agent_indices[0]) == k
    losses = sq.loss(mo, po, metas, {"indices": indices})
    assert torch.isfinite(losses["distribution_loss"])
    inv = U.build_inverse_index(sq.keep_idx, N_A)
    kept, _ = U.remap_match_indices(indices, inv)
    assert 0 < len(kept[0][0]) < n  # some positives were pruned and dropped from the motion loss


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
