"""Analytic cost model for Sparse Query pruning (DistillDrive student, nuScenes stage2 config).

    python tools/sparse_query_cost_model.py

Counts multiply-accumulates (MACs) of every matmul / conv on the planning path
(paper Fig. 2: Temporal / Agent / Map Decoder x3, Generative Decoder, IL-Guide Head)
as a function of the number of agent queries n that enter it:

    C(n) = alpha * n + beta          (n = N_A = 900 before, n = K_eff after)

All dimensions are read off projects/configs/stage2/distilldrive_stage2_distribution.py.
Paper notation: N_A agents, N_E ego modes, T history frames, D feature dim.
"""
from dataclasses import dataclass, field


@dataclass
class Cfg:
    D: int = 256            # embed_dims
    N_A: int = 900          # det num_anchor
    N_E: int = 18           # 3 commands x ego_fut_mode 6
    T: int = 4              # queue_length (Memory Bank frames)
    num_det: int = 50       # Agent Decoder keys (top-k of N_A)
    num_map: int = 10       # Map Decoder keys (top-k of N_L)
    blocks: int = 3         # [temp_gnn, gnn, cross_gnn, ffn] x 3 in the student head
    decouple: bool = True   # decouple_attn_motion: temp_gnn / gnn run at 2D with fc_before/after
    ffn_hidden_mult: int = 2
    fut_mode: int = 6
    fut_ts: int = 12
    ego_fut_ts: int = 6
    latent: int = 32        # distribution latent_dim
    gru_layers: int = 4     # layer_dim
    plan_instance_dim: int = 128
    anchor_encoder: str = "sparsebox3d"  # "sparsebox3d" (real) | "linear11" (test harness)


def anchor_encoder_macs(c: Cfg) -> int:
    if c.anchor_encoder == "linear11":
        return 11 * c.D
    # SparseBox3DEncoder, decouple mode: cat of [128, 32, 32, 64], out_loops = 4, no output_fc
    def emb(i, o):
        return i * o + 3 * o * o
    return emb(3, 128) + emb(3, 32) + emb(2, 32) + emb(3, 64)


def per_token(c: Cfg) -> dict:
    """MACs for ONE query token (agent or ego) through the decoder stack."""
    D, T = c.D, c.T
    A = 2 * D if c.decouple else D          # attention width
    fc_b = D * A if c.decouple else 0       # fc_before (value D -> 2D)
    fc_a = A * D if c.decouple else 0       # fc_after  (2D -> D)
    kd = c.num_det + c.N_E

    # Temporal Decoder: every query owns T history keys -> k/v projections scale with n*T
    temporal = (A * A          # q in-proj
                + T * A * A    # k in-proj on T keys
                + T * A * A    # v in-proj on T values
                + T * fc_b     # fc_before on T values
                + 2 * T * A    # scores + weighted sum
                + A * A        # out-proj
                + fc_a)        # fc_after
    agent = A * A + 2 * kd * A + A * A + fc_a                 # keys shared -> fixed part below
    mapd = D * D + 2 * c.num_map * D + D * D                  # not decoupled
    ffn = 2 * D * (c.ffn_hidden_mult * D)
    per_block = temporal + agent + mapd + ffn
    enc = T * anchor_encoder_macs(c)                           # temp_anchor_embed on T anchors

    # Generative Decoder: DistributionModule(D -> latent) + GRU(latent, 2*latent, L) + MLP
    cd = D // 2
    present = D * 2 * D + 2 * D * 2 * D + 2 * D * cd + cd * 2 * c.latent
    H = 2 * c.latent
    gru = 3 * (c.latent * H + H * H) + (c.gru_layers - 1) * 3 * (H * H + H * H)
    mlp = H * 2 * H + 2 * H * 4 * H + 4 * H * D
    return dict(temporal=c.blocks * temporal, agent=c.blocks * agent, map=c.blocks * mapd,
                ffn=c.blocks * ffn, anchor_enc=enc, gen_conv=present, gen_gru=gru + mlp,
                total=c.blocks * per_block + enc + present + gru + mlp)


def per_agent_refine(c: Cfg) -> dict:
    """IL-Guide Head motion part for one agent (all modes)."""
    D, M = c.D, c.fut_mode
    mode_pos = 2 * D * D                        # motion_anchor_encoder: Linear + Linear
    cls = 2 * D * D + D                         # linear_relu_ln(D,1,2) + Linear(D,1)
    reg = 2 * D * D + D * c.fut_ts * 2
    return dict(refine=M * (mode_pos + cls + reg))


def fixed(c: Cfg) -> dict:
    """Terms that do not depend on n."""
    D = c.D
    A = 2 * D if c.decouple else D
    kd = c.num_det + c.N_E
    agent_kv = c.blocks * kd * (2 * A * A + (D * A if c.decouple else 0))
    map_kv = c.blocks * c.num_map * 2 * D * D
    ego = c.N_E * per_token(c)["total"] + c.N_E * anchor_encoder_macs(c)  # ego tokens + ego_anchor_embed
    pid = c.plan_instance_dim * c.ego_fut_ts
    plan_inst = c.N_E * (pid * pid + pid * D)          # plan_instance_encoder
    plan_pos = c.N_E * 2 * D * D                       # plan_anchor_encoder
    plan_heads = c.N_E * (2 * D * D + D) + c.N_E * (2 * D * D + D * c.ego_fut_ts * 2) + (2 * D * D + 10 * D)
    return dict(agent_kv=agent_kv, map_kv=map_kv, ego_tokens=ego, planning=plan_inst + plan_pos + plan_heads)


def planner_macs(n: int, c: Cfg) -> dict:
    t = per_token(c)["total"] + per_agent_refine(c)["refine"]
    f = sum(fixed(c).values())
    return dict(alpha=t, beta=f, total=n * t + f)


def memory_bank_match(c: Cfg, B: int = 1) -> dict:
    """InstanceQueue.prepare_motion: (match[..., None] * feat[:, None]).sum(2) per queue entry.
    Materialises [B, N_A, N_A, D] -- independent of K (Memory Bank keeps N_A slots)."""
    elems = B * c.N_A * c.N_A * c.D
    return dict(elementwise_ops=2 * elems * c.T, bytes_per_entry=4 * elems,
                traffic_bytes=c.T * 2 * 4 * elems)  # write + read of the product, T entries


# ── rest of the model (for Amdahl) ──────────────────────────────────────────
def perception_macs(c: Cfg, n_det=900, n_temp=600, n_map=100, layers=6) -> dict:
    D = c.D
    A = 2 * D  # decouple_attn = True in the det head
    gnn = 3 * A * A + D * A + A * A + A * D + 2 * n_det * A           # self-attn, per query
    tmp = A * A + A * A + A * D + 2 * n_temp * A                       # temporal cross-attn, per query
    tmp_fixed = n_temp * (2 * A * A + D * A)
    dfa = D * (8 * 4 * 6 * 13) + 312 * D * 5 + D * D                   # weights_fc + sampling + out proj
    ffn = 2 * D * 4 * D + 4 * D * D + 2 * D * D                        # 512->1024->256 + identity 512->256
    refine = 8 * D * D
    enc = anchor_encoder_macs(c)
    det = n_det * (layers * (gnn + dfa + ffn + refine + enc) + (layers - 1) * tmp) + (layers - 1) * tmp_fixed
    mp = n_map * layers * (4 * D * D + 2 * n_map * D + D * (8 * 4 * 6 * 23) + 23 * 4 * 6 * D * 5 + 4 * D * D + 8 * D * D)
    return dict(det=det, map=mp)


def fmt(x):
    return f"{x / 1e9:7.2f} G"


if __name__ == "__main__":
    c = Cfg()
    D2 = c.D * c.D
    pt, rf, fx = per_token(c), per_agent_refine(c), fixed(c)
    print("per agent query (units of D^2 = 65,536 MACs)")
    for k, v in {**pt, **rf}.items():
        print(f"  {k:10s} {v / D2:8.2f} D^2")
    pm = planner_macs(0, c)
    print(f"alpha = {pm['alpha'] / D2:.1f} D^2 = {pm['alpha'] / 1e6:.2f} M MACs per agent")
    print(f"beta  = {pm['beta'] / D2:.0f} D^2 = {pm['beta'] / 1e9:.2f} G MACs fixed  {({k: round(v / D2) for k, v in fx.items()})}")
    base = planner_macs(c.N_A, c)["total"]
    print("\nplanning path total")
    for n in [900, 300, 150, 110, 64, 32]:
        t = planner_macs(n, c)["total"]
        print(f"  n={n:4d}  {fmt(t)}   x{base / t:5.2f}   -{100 * (1 - t / base):4.1f}%")
    mb = memory_bank_match(c)
    print(f"\nMemory Bank match (B=1): {mb['bytes_per_entry'] / 1e6:.0f} MB per entry, "
          f"{mb['traffic_bytes'] / 1e9:.1f} GB traffic / frame, unchanged by K")
    pr = perception_macs(c)

    # ── end-to-end FLOPs (inference, 6 cameras, 256 x 704) ──
    # ResNet-50 and FPN measured with torch FlopCounterMode (torchvision modules, same shapes)
    parts = {"ResNet-50 x6": 88.1e9, "FPN x6": 61.2e9,
             "Perception det (est.)": pr["det"], "Perception map (est.)": pr["map"]}
    rest = sum(parts.values())
    print("\nend-to-end MACs")
    for k, v in parts.items():
        print(f"  {k:24s} {fmt(v)}")
    print(f"  {'planning path':24s} {fmt(base)}   share p = {base / (rest + base):.1%}")
    for n in [110, 64]:
        t = planner_macs(n, c)["total"]
        tot0, tot1 = rest + base, rest + t
        p, s = base / tot0, base / t
        print(f"  n={n:3d}: total {fmt(tot0)} -> {fmt(tot1)}  (-{100 * (1 - tot1 / tot0):.1f}%)  "
              f"Amdahl 1/((1-p)+p/s) = x{1 / ((1 - p) + p / s):.3f}")

    # ── GPU latency model for the planning head (assumptions, not measurements) ──
    print("\nGPU model: t_head(n) = t_fixed + t_MB + 2*C(n)/F_eff")
    for gpu, bw in [("A800 (2.0 TB/s)", 2.0e12), ("RTX 4090 (1.0 TB/s)", 1.0e12)]:
        print(f"  {gpu:20s} t_MB >= {mb['traffic_bytes'] / bw * 1e3:.1f} ms (bandwidth bound, K-independent)")
    for F in [20e12, 60e12]:
        d = 2 * (base - planner_macs(64, c)["total"]) / F
        print(f"  F_eff {F / 1e12:.0f} TFLOPS: GEMM time saved by K=64 = {d * 1e3:.2f} ms "
              f"= {d / (1 / 6.0):.1%} of a 6.0 FPS frame (paper Table 1)")
