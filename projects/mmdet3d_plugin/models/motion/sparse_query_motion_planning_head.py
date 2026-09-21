"""Sparse Query pruning for the DistillDrive planning decoder (paper Sec. 3.3).

Shrinks the agent part N_A of the query tensor that runs through the
Temporal Decoder -> Agent Decoder -> Map Decoder (x3) -> Generative Decoder
-> Multi-mode Planning, from N_A + N_E = 900 + 18 down to K + 18.

Left untouched on purpose:
  * Perception Model (det/map heads) : still produces N_A = 900 instances.
  * Memory Bank (InstanceQueue)      : still tracks all 900 slots by instance_id.
  * Agent / Map Decoder keys          : still top-`num_det` / top-`num_map` of the
                                        full set, so every kept query sees exactly
                                        the same keys as in the unpruned model.
  * Multi-mode Planning outputs       : motion predictions are scattered back to
                                        900 slots, so losses and decoders run as is.

`num_query=None` (or >= N_A) reproduces the parent class exactly.

All changed lines relative to MotionPlanningHead.forward are tagged `# [SQ]`.
"""
import torch

from mmdet.models import HEADS

from .motion_planning_head import MotionPlanningHead
from ..attention import gen_sineembed_for_position
from ..instance_bank import topk
from .sparse_query_utils import (
    append_ego_slots,
    build_inverse_index,
    gather_tokens,
    gru_hidden_original,
    gru_hidden_per_token,
    gru_hidden_preserve,
    gru_source_mask,
    gt_proximity_mask,
    probabilistic_loss_pruned,
    remap_match_indices,
    scatter_tokens,
    select_queries,
)

_GEN_MODES = ("original", "per_token", "preserve")


@HEADS.register_module()
class SparseQueryMotionPlanningHead(MotionPlanningHead):
    """
    Args (in addition to MotionPlanningHead):
        num_query (int | None): K, number of agent queries kept. None = no pruning.
        min_keep_score (float | None): agents with det confidence >= this are always
            kept. 0.5 matches HierarchicalPlanningDecoder.rescore(score_thresh=0.5),
            so collision rescoring sees the same agents as the unpruned model.
        generative_hidden (str): how the Generative Decoder builds the GRU h0.
            'preserve'  - reproduce the original token-mixing reshape for ego rows
                          (and rescore agents), force-keeping their source slots.
                          Use with released stage2 weights, no training.
            'original'  - apply the original reshape to the pruned tensor. Ego h0
                          then reads different agent slots than at training time.
            'per_token' - per-token split. Independent of K and batch; changes the
                          model, so fine-tune before evaluating.
        preserve_rescore (bool): in 'preserve' mode, also make the GRU h0 of the
            min_keep_score agents exact (needed for bit-exact final planning).
        force_positive (bool): at train time, force-keep the anchor nearest to each
            GT box so motion / generative supervision is not pruned away.
        force_positive_radius (float): BEV radius in metres for that match.
        force_positive_prob (float): probability of applying it per iteration
            (anneal 1.0 -> 0.0 to close the train/test gap if desired).
        pruned_cls_fill (float): logit written into pruned motion slots.
            sigmoid(-60) ~ 0, so decoders read "no prediction".
    """

    def __init__(
        self,
        num_query=None,
        min_keep_score=0.5,
        generative_hidden="preserve",
        preserve_rescore=True,
        force_positive=True,
        force_positive_radius=2.0,
        force_positive_prob=1.0,
        pruned_cls_fill=-60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert generative_hidden in _GEN_MODES, generative_hidden
        if isinstance(num_query, str):  # --cfg-options may pass "None" as a string
            num_query = None if num_query.lower() == "none" else int(num_query)
        self.num_query = num_query
        self.min_keep_score = min_keep_score
        self.generative_hidden = generative_hidden
        self.preserve_rescore = preserve_rescore
        self.force_positive = force_positive
        self.force_positive_radius = force_positive_radius
        self.force_positive_prob = force_positive_prob
        self.pruned_cls_fill = pruned_cls_fill
        if generative_hidden != "original" and self.vae_model is True:
            assert not self.auto_regression, "per_token/preserve support temporal_frames non-AR only"
        self._reset_sq_state()

    # ------------------------------------------------------------------ state
    def _reset_sq_state(self):
        self.keep_idx = None          # [B, K] kept agent slots (sorted), or None
        self.num_anchor_full = None   # N_A before pruning
        self.sq_stats = {}            # last-forward diagnostics (k_eff, forced counts)

    def _sq_active(self):
        return self.keep_idx is not None

    # --------------------------------------------------------------- selection
    def _build_must_keep(self, det_confidence, det_anchors, metas, num_ego):
        """[B, N_A] bool of slots that must survive pruning."""
        B, N = det_confidence.shape
        must = torch.zeros_like(det_confidence, dtype=torch.bool)
        stats = {}

        if self.min_keep_score is not None:
            score_keep = det_confidence >= self.min_keep_score
            must |= score_keep
            stats["n_score_keep"] = int(score_keep.sum(1).max())
        else:
            score_keep = torch.zeros_like(must)

        if (
            self.training
            and self.force_positive
            and "gt_bboxes_3d" in metas
            and (self.force_positive_prob >= 1.0 or torch.rand(()).item() < self.force_positive_prob)
        ):
            pos = gt_proximity_mask(det_anchors[..., :2], metas["gt_bboxes_3d"], self.force_positive_radius)
            must |= pos
            stats["n_positive"] = int(pos.sum(1).max())

        if self.vae_model is True and self.generative_hidden == "preserve":
            rows = torch.zeros(B, N + num_ego, dtype=torch.bool, device=must.device)
            rows[:, N:] = True                          # ego rows
            if self.preserve_rescore:
                rows[:, :N] = score_keep                # rescore agents' rows
            src = gru_source_mask(rows, N, self.layer_dim)
            must |= src
            stats["n_gru_source"] = int(src.sum(1).max())

        return must, stats

    # --------------------------------------------------------- generative step
    def _generative_decoder(self, instance_feature, future_state, fallback_agent):
        """Generative Decoder (Eq. 5): distribution sampling + GRU fusion."""
        if not self._sq_active() or self.generative_hidden == "original":
            sample, dist = self.distribution_forward(instance_feature, future_state)
            feat, _ = self.future_states_predict(
                sample=sample, hidden_states=instance_feature, current_states=instance_feature
            )
            return feat, dist

        noise = None
        if self.generative_hidden == "preserve":
            # Draw noise for the full layout and keep the kept rows, so ego rows get the
            # same random numbers as the unpruned model under the same RNG state.
            # The parent calls randn_like(mu) where mu = conv_out.permute(0, 2, 1)[..., :LD];
            # randn_like keeps that (channel-major, when dense) memory layout, so the
            # template must have the same strides or the values land in other positions.
            B, KE, _ = instance_feature.shape
            num_ego = KE - self.keep_idx.shape[1]
            n1f = self.num_anchor_full + num_ego
            template = torch.empty(
                (B, 2 * self.latent_dim, n1f), dtype=instance_feature.dtype, device=instance_feature.device
            ).permute(0, 2, 1)[:, :, : self.latent_dim]
            full = torch.randn_like(template)
            noise = gather_tokens(full, append_ego_slots(self.keep_idx, self.num_anchor_full, num_ego))

        sample, dist = self.distribution_forward(instance_feature, future_state, noise=noise)

        # == future_states_predict with a switchable h0 builder (non-AR path) ==
        BS, N1 = instance_feature.shape[:2]
        fpi = sample.unsqueeze(0).expand(self.temporal_frames, -1, -1, -1).permute(0, 1, 3, 2).contiguous()
        fpi = fpi.reshape(self.temporal_frames, -1, self.latent_dim)  # [T, B*N1, LD]
        if self.generative_hidden == "per_token":
            h0 = gru_hidden_per_token(instance_feature, self.layer_dim)
        else:
            h0 = gru_hidden_preserve(instance_feature, self.keep_idx, fallback_agent, self.layer_dim)
        fs = self.predict_model(fpi, h0.contiguous())  # [T, B*N1, D]
        fs = fs.reshape(BS, N1, fs.shape[2])
        feat = instance_feature + fs if self.with_cur else fs
        return feat, dist

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):
        self._reset_sq_state()
        num_ego = self.ego_fut_mode * 3

        # =========== agent/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]  # [B, N, D]
        anchor_embed = det_output["anchor_embed"]  # [B, N, D]
        det_classification = det_output["classification"][-1].sigmoid()  # [B, N, 10]
        det_anchors = det_output["prediction"][-1]  # [B, N, 11]
        det_confidence = det_classification.max(dim=-1).values  # [B, N]
        # Agent Decoder keys: top-num_det of the FULL set (unchanged from parent)
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        bs, num_anchor, dim = instance_feature.shape
        fallback_agent = instance_feature.detach()  # [SQ] pre-decoder features for 'preserve'

        # [SQ] ================= sparse query selection =================
        if self.num_query is not None and self.num_query < num_anchor:
            must, stats = self._build_must_keep(det_confidence, det_anchors, metas, num_ego)
            keep_idx, k_eff = select_queries(det_confidence, self.num_query, must)
            if k_eff < num_anchor:
                self.keep_idx = keep_idx
                self.num_anchor_full = num_anchor
                instance_feature = gather_tokens(instance_feature, keep_idx)
                anchor_embed = gather_tokens(anchor_embed, keep_idx)
                det_classification = gather_tokens(det_classification, keep_idx)
                det_anchors = gather_tokens(det_anchors, keep_idx)
                num_anchor = k_eff
            stats.update(k_eff=k_eff, num_query=self.num_query, n_full=self.num_anchor_full or num_anchor)
            self.sq_stats = stats

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)  # [B, K, MA, T, 2]
        plan_anchor = torch.tile(self.plan_anchor[None], (bs, 1, 1, 1, 1))
        plan_anchor_delta = torch.tile(self.plan_anchor_delta[None], (bs, 1, 1, 1, 1))

        # =========== mode endpoint pos embed init ===========
        motion_mode_pos = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_pos = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)

        # =========== plan trajectory instance feature ===========
        plan_instance_pos = gen_sineembed_for_position(plan_anchor_delta, 128)
        plan_mode_query = self.plan_instance_encoder(plan_instance_pos.flatten(-2)).flatten(1, 2).unsqueeze(1)

        # ========== Memory Bank: runs on the FULL det_output (900 slots) ==========
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
            plan_mode_query,
        )
        # [SQ] Temporal Decoder input: keep only the rows of kept agents + ego
        if self._sq_active():
            rows = append_ego_slots(self.keep_idx, self.num_anchor_full, num_ego)
            temp_instance_feature = gather_tokens(temp_instance_feature, rows)
            temp_anchor = gather_tokens(temp_anchor, rows)
            temp_mask = gather_tokens(temp_mask, rows)

        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)
        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)
        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)
        ego_feature_memory = ego_feature.clone()

        if self.training:
            future_state = self.get_future_state(
                metas["gt_ego_fut_trajs"],
                metas["gt_ego_fut_masks"],
                metas["gt_agent_fut_trajs"],
                metas["gt_agent_fut_masks"],
                metas["gt_labels_3d"],
            )  # [B, K + N_E, T * 2]   ([SQ] length follows num_anchor via override)
        else:
            future_state = None

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        planning_feature = []
        planning_memory_feature = []
        output_distribution = None
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                if op == "distribution":
                    # [SQ] Generative Decoder with switchable GRU h0
                    instance_feature, output_distribution = self._generative_decoder(
                        instance_feature, future_state, fallback_agent
                    )
                else:
                    continue
            elif op == "temp_gnn":  # Temporal Decoder
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + num_ego, dim)
            elif op == "gnn":  # Agent Decoder
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
                if op == "ffn":
                    planning_memory_feature.append(instance_feature[:, num_anchor:])
            elif op == "cross_gnn":  # Map Decoder
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":  # Multi-mode Planning
                motion_query = motion_mode_pos + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_pos + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(1)
                ego_feature = self.ego_feature_avp(instance_feature[:, num_anchor:].permute(0, 2, 1)).permute(0, 2, 1)
                ego_anchor_embed = self.ego_pos_avp(anchor_embed[:, num_anchor:].permute(0, 2, 1)).permute(0, 2, 1)
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    ego_feature,
                    ego_anchor_embed,
                )
                if self.pred_delta:
                    plan_delta = plan_anchor_delta.clone().flatten(1, 2).unsqueeze(1)
                    plan_reg = plan_reg + plan_delta
                # [SQ] scatter motion back to N_A slots -> losses/decoders unchanged
                if self._sq_active():
                    motion_cls = scatter_tokens(motion_cls, self.keep_idx, self.num_anchor_full, self.pruned_cls_fill)
                    motion_reg = scatter_tokens(motion_reg, self.keep_idx, self.num_anchor_full, 0.0)
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
                planning_feature.append(plan_query)

        # first arg is unused by InstanceQueue.cache_motion; queue stays at N_A slots
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "feature": planning_feature,
            "encoder_feature": ego_feature_memory,
            "decoder_feature": planning_memory_feature,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        if self.vae_model is True:
            planning_output["distribution"] = output_distribution

        return motion_output, planning_output

    # ------------------------------------------------------ Generative targets
    def get_future_state(self, gt_ego_fut_trajs, gt_ego_fut_masks, gt_agent_fut_trajs, gt_agent_fut_masks, gt_labels_3d):
        """Parent hard-codes agent_dim = 900. Match it to the pruned length instead."""
        if not self._sq_active():
            return super().get_future_state(
                gt_ego_fut_trajs, gt_ego_fut_masks, gt_agent_fut_trajs, gt_agent_fut_masks, gt_labels_3d
            )

        agent_dim = self.keep_idx.shape[1]
        veh_list = [0, 1, 3, 4, 8]  # same as parent: car, truck, bus, trailer, pedestrian
        self.agent_indices = []
        rows = []
        n_truncated = 0
        for b in range(len(gt_labels_3d)):
            labels = gt_labels_3d[b]
            trajs = gt_agent_fut_trajs[b]
            dev = gt_agent_fut_masks[b].device
            veh = torch.tensor([int(l) in veh_list for l in labels], dtype=torch.bool, device=dev)
            idx = torch.where(veh)[0]
            n_truncated += max(0, len(idx) - agent_dim)
            idx = idx[:agent_dim]  # [SQ] only K future slots exist
            self.agent_indices.append(idx)
            fut = torch.zeros([agent_dim, self.ego_fut_ts, 2], device=dev)
            if len(idx) > 0:
                fut[: len(idx)] = trajs[idx][:, : self.ego_fut_ts, :]
            rows.append(fut)
        gt_trajs = torch.cat(
            (torch.stack(rows), gt_ego_fut_trajs.unsqueeze(1).repeat(1, self.ego_fut_mode * 3, 1, 1)),
            dim=1,
        )  # [B, K + N_E, T, 2]
        self.sq_stats["n_future_truncated"] = n_truncated
        return gt_trajs.flatten(-2, -1)

    # -------------------------------------------------------------------- loss
    def loss(self, motion_model_outs, planning_model_outs, data, motion_loss_cache):
        if not self._sq_active():
            return super().loss(motion_model_outs, planning_model_outs, data, motion_loss_cache)

        inv = build_inverse_index(self.keep_idx, self.num_anchor_full)
        kept_full, mapped_k = remap_match_indices(motion_loss_cache["indices"], inv)
        # diagnostics: Hungarian positives lost to pruning (force_positive uses BEV
        # nearest anchor, which is not always the slot the cls+box cost matched)
        n_match = sum(len(p) for p, _ in motion_loss_cache["indices"] if p is not None)
        n_kept = sum(len(p) for p, _ in kept_full if p is not None)
        self.sq_stats["n_pos_dropped"] = n_match - n_kept
        self.sq_stats["n_pos_total"] = n_match

        loss = {}
        # Multi-mode Planning (motion): outputs are in N_A slots; drop pruned matches
        cache = dict(motion_loss_cache)
        cache["indices"] = kept_full
        loss.update(self.loss_motion(motion_model_outs, data, cache))
        loss.update(self.loss_planning(planning_model_outs, data))

        # Generative Decoder (Eq. 6)
        if self.vae_model is True:
            gen = self.loss_vae_gen
            if getattr(gen, "only_valid_agent", False):
                loss["distribution_loss"] = probabilistic_loss_pruned(
                    planning_model_outs["distribution"],
                    mapped_k,
                    self.agent_indices,
                    ego_mode_num=gen.ego_mode_num,
                    loss_weight=gen.loss_weight,
                )
            else:
                loss.update(gen(planning_model_outs["distribution"], motion_loss_cache, self.agent_indices))
        return loss
