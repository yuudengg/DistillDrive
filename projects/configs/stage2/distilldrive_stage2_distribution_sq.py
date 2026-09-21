# Sparse Query pruning on top of the released stage2 student config.
# Only the planning decoder's query tensor (N_A + N_E) is shrunk; see
# projects/mmdet3d_plugin/models/motion/sparse_query_motion_planning_head.py
#
# ── Phase 0: released stage2 weights, no training ─────────────────────────────
# (A) sanity run — planning must equal the unpruned baseline (up to ~1e-6):
#   bash ./tools/dist_test.sh projects/configs/stage2/distilldrive_stage2_distribution_sq.py \
#        ckpt/distilldrive_stage2_distribution.pth 8 --deterministic --eval bbox
#   Effective K is floored at ~15 + 5 x (#agents with conf >= 0.5), ~100 on nuScenes.
#   Read the real value from head.sq_stats["k_eff"], not from num_query.
#
# (B) K sweep — ego stays exact, rescore agents may differ slightly:
#   for K in 150 100 64 32; do
#     bash ./tools/dist_test.sh <this cfg> <ckpt> 8 --deterministic --eval bbox \
#          --cfg-options model.head.motion_plan_head.num_query=$K \
#                        model.head.motion_plan_head.preserve_rescore=False
#   done
#   Metric to watch: motion forecasting (EPA / minADE / minFDE / MR). Planning
#   L2 / collision should stay flat; pruned low-confidence boxes get a stationary
#   trajectory with score ~0.
#
# ── Phase 1: fine-tune ───────────────────────────────────────────────────────
#   generative_hidden="per_token", load_from=<stage2 ckpt>. Removes the GRU h0
#   token mixing, so K can go below the ~100 floor. Also fine-tune a
#   per_token baseline (num_query=900) for a fair comparison.
#
# To disable pruning from the command line use num_query=900 (not None).

_base_ = ["./distilldrive_stage2_distribution.py"]

num_query = 64

model = dict(
    head=dict(
        motion_plan_head=dict(
            type="SparseQueryMotionPlanningHead",
            num_query=num_query,
            min_keep_score=0.5,            # = HierarchicalPlanningDecoder.rescore score_thresh
            generative_hidden="preserve",  # "preserve" | "original" | "per_token"
            preserve_rescore=True,
            force_positive=True,           # train only
            force_positive_radius=2.0,
            force_positive_prob=1.0,
            pruned_cls_fill=-60.0,
        )
    )
)
