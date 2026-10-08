from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.diffusion import DistributionModule, PredictModel

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


@HEADS.register_module()
class MotionPlanningHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        pred_delta=False,
        select_command_kd=False,
        adapter_tag=False,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        distribution_cfg=None,
        dqn_cfg=None,
        multi_modal_cfg=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        loss_vae_gen=None,
        kd_loss_cls=None,
        kd_loss_reg=None,
        kd_loss_feats=None,
        kd_loss_en_feats=None,
        kd_loss_de_feats=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
    ):

        super(MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.pred_delta = pred_delta
        self.select_command_kd = select_command_kd
        self.adapter_tag = adapter_tag

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # distribution model
        if distribution_cfg is not None:
            self.with_cur = distribution_cfg.with_cur
            self.latent_dim = distribution_cfg.latent_dim
            self.layer_dim = distribution_cfg.layer_dim
            self.future_concate = distribution_cfg.future_concate # while concate present feature
            self.log_paramter = distribution_cfg.log_paramter
            self.temporal_frames = distribution_cfg.temporal_frames
            self.auto_regression = distribution_cfg.get('auto_regression', False)
            if self.auto_regression is True:
                assert self.temporal_frames == 6, "Auto regression must have time dimension as 6"
                assert self.fut_ts == self.ego_fut_ts, "Agent future steps must same as Ego"
            self.present_distribution = DistributionModule(
                distribution_cfg.present_distribution_in_channels,
                distribution_cfg.latent_dim,
                min_log_sigma=distribution_cfg.min_log_sigma,
                max_log_sigma=distribution_cfg.max_log_sigma,
            )

            self.future_distribution = DistributionModule(
                distribution_cfg.future_distribution_in_channels,
                distribution_cfg.latent_dim,
                min_log_sigma=distribution_cfg.min_log_sigma, 
                max_log_sigma=distribution_cfg.max_log_sigma,
            )

            self.predict_model = PredictModel(
                in_channels=distribution_cfg.latent_dim,
                out_channels=self.embed_dims,
                hidden_channels=distribution_cfg.latent_dim * 2,
                num_layers=distribution_cfg.layer_dim,
                temporal_frames=distribution_cfg.temporal_frames,
            )
            self.vae_model = True

        else:
            self.vae_model = None
        self.loss_vae_gen = build_loss(loss_vae_gen)

        # Reinforcement learning model
        if dqn_cfg is not None:
            self.dqn_model = build(dqn_cfg, PLUGIN_LAYERS)
        else:
            self.dqn_model = None

        # Knowledge distillation setting
        if kd_loss_cls is not None:
            self.kd_loss_cls = build_loss(kd_loss_cls)
        else:
            self.kd_loss_cls = None
        if kd_loss_reg is not None:
            self.kd_loss_reg = build_loss(kd_loss_reg)
        else:
            self.kd_loss_reg = None
        if kd_loss_feats is not None:
            self.kd_loss_feats = build_loss(kd_loss_feats)   
        else:
            self.kd_loss_feats = None  
        if kd_loss_en_feats is not None:
            self.kd_loss_en_feats = build_loss(kd_loss_en_feats)   
        else:
            self.kd_loss_en_feats = None  
        if kd_loss_de_feats is not None:
            self.kd_loss_de_feats = build_loss(kd_loss_de_feats)   
        else:
            self.kd_loss_de_feats = None 
   
        # Adpater for knowdledge distillation
        if self.adapter_tag:
            self.feature_adapter = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )

        # multi modal setting
        if multi_modal_cfg is not None:
            self.multi_modal = True
            self.plan_instance_dim = multi_modal_cfg.plan_instance_dim
            self.ego_feature_avp = nn.AdaptiveAvgPool1d(1)
            self.ego_pos_avp = nn.AdaptiveAvgPool1d(1)
        else:
            self.multi_modal = False
            self.plan_instance_dim = 128

        # motion init anchor and encoder
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan init anchor and encoder
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # planning instance and anchor delta
        plan_anchor_delta = np.copy(plan_anchor)
        plan_anchor_delta[:, :, 1:, :] = plan_anchor_delta[:, :, 1:, :] - plan_anchor_delta[:, :, :-1, :]
        self.plan_anchor_delta = nn.Parameter(
            torch.tensor(plan_anchor_delta, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_instance_encoder = nn.Sequential(
            *linear_relu_ln(self.plan_instance_dim * self.ego_fut_ts, 1, 1),
            Linear(self.plan_instance_dim * self.ego_fut_ts, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

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
        # =========== agent/map feature/anchor ===========
        instance_feature = det_output["instance_feature"] # [B, N, D]
        anchor_embed = det_output["anchor_embed"] # [B, N, D]
        det_classification = det_output["classification"][-1].sigmoid() # [B, N, 10]
        det_anchors = det_output["prediction"][-1] # [B, N, 11]
        det_confidence = det_classification.max(dim=-1).values # [B, N]
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )

        map_instance_feature = map_output["instance_feature"] # [B, N_M, D]
        map_anchor_embed = map_output["anchor_embed"] # [B, N_M, D]
        map_classification = map_output["classification"][-1].sigmoid() # [B, N_M, 3]
        map_anchors = map_output["prediction"][-1] # [B, N_M, 40]
        map_confidence = map_classification.max(dim=-1).values # [B, N_M]
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        bs, num_anchor, dim = instance_feature.shape
        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors) # [B, N, MA, T, 2]
        plan_anchor = torch.tile(
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        ) # [B, 3, ME, T, 2]
        plan_anchor_delta = torch.tile(
            self.plan_anchor_delta[None], (bs, 1, 1, 1, 1)
        ) # [B, 3, ME, T, 2]

        # =========== mode endpoint pos embed init ===========
        motion_mode_pos = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :])) # [B, N, MA, D]
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :]) # [B, 3, ME, D]
        plan_mode_pos = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1) # [B, 1, 3 * ME, D]

        # =========== plan trajectory insatnce feature ===========
        plan_instance_pos = gen_sineembed_for_position(plan_anchor_delta, 128) # [B, 3, ME, T, D]
        plan_mode_query = self.plan_instance_encoder(plan_instance_pos.flatten(-2)).flatten(1, 2).unsqueeze(1) # [B, 1, 3 * ME, D]

        # ========== get ego/temporal feature/anchor ===========
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
        ego_anchor_embed = anchor_encoder(ego_anchor) # [B, ME, D]
        temp_anchor_embed = anchor_encoder(temp_anchor) # [B, N1, T, D]
        temp_instance_feature = temp_instance_feature.flatten(0, 1) # [B * N1, T, D]
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1) # [B * N1, T, D]
        temp_mask = temp_mask.flatten(0, 1) # [B * N1, T]
        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)
        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)
        ego_feature_memory = ego_feature.clone()
        # training stage
        if self.training:
            gt_ego_fut_trajs = metas['gt_ego_fut_trajs'] # [B, T1, 2]
            gt_ego_fut_masks = metas['gt_ego_fut_masks'] # [B, T1]
            gt_agent_fut_trajs = metas['gt_agent_fut_trajs'] # B *[A, T2, 2]
            gt_agent_fut_masks = metas['gt_agent_fut_masks'] # B *[A, T2]
            gt_labels_3d = metas['gt_labels_3d'] # B *[A, 1]

            future_state = self.get_future_state(
                gt_ego_fut_trajs, gt_ego_fut_masks, gt_agent_fut_trajs, gt_agent_fut_masks, gt_labels_3d
            ) # [B, N1 + M_E, T * 2]
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
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                if op == "distribution":
                    sample, output_distribution = self.distribution_forward(
                        instance_feature, future_state
                    ) # [B, D1, N1 + ME], Dict
                    instance_feature, future_states_hs = self.future_states_predict(
                        sample=sample,
                        hidden_states=instance_feature,
                        current_states=instance_feature
                    ) # [B, N1 + ME, D]
                else:
                    continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + self.ego_fut_mode * 3, dim) # [B, N1 + ME, D]
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                ) # [B, N1 + ME, D]
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
                if op == "ffn":
                    planning_memory_feature.append(instance_feature[:, num_anchor:])
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )  # [B, N1 + M_E, D]
            elif op == "refine":
                # 1st pass (cond=None): identical to the original refine block
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                    plan_query,
                ) = self._refine_once(
                    i,
                    instance_feature,
                    anchor_embed,
                    num_anchor,
                    motion_mode_pos,
                    plan_mode_pos,
                    plan_anchor_delta,
                    cond=None,
                )
                # TODO(STEP 2): feedback 2nd pass, refine only (no gnn / VAE re-run)
                # if self.use_feedback:
                #     cond = self.adaln(...)  # meta-action -> condition
                #     (...) = self._refine_once(..., cond=cond)
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
                planning_feature.append(plan_query)

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
        # Add distribution to calculate the gradient
        if self.vae_model is True:
            planning_output["distribution"] = output_distribution

        return motion_output, planning_output

    def _refine_once(
        self,
        i,
        instance_feature,
        anchor_embed,
        num_anchor,
        motion_mode_pos,
        plan_mode_pos,
        plan_anchor_delta,
        cond=None,
    ):
        """Run the refine block once (plan B: refine only, no gnn / VAE).

        cond is reserved for the feedback (adaLN) 2nd pass and is unused for now,
        so with cond=None the output is identical to the original refine block.
        """
        # get motion and planing feature
        motion_query = motion_mode_pos + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2) # [B, N1, MA, D]
        plan_query = plan_mode_pos + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(1) # [B, 1, ME, D]
        # Only get one Status, Must to commperss in MultiModal
        ego_feature = self.ego_feature_avp(instance_feature[:, num_anchor:].permute(0, 2, 1)).permute(0, 2, 1) # [B, 1, D]
        ego_anchor_embed = self.ego_pos_avp(anchor_embed[:, num_anchor:].permute(0, 2, 1)).permute(0, 2, 1)  # [B, 1, D]
        # Need to update Memory Bank
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
        # anchor + offset schedule
        if self.pred_delta:
            plan_delta = plan_anchor_delta.clone().flatten(1, 2).unsqueeze(1) # [B, 1, ME, T2, 2]
            plan_reg = plan_reg + plan_delta
        return motion_cls, motion_reg, plan_cls, plan_reg, plan_status, plan_query

    def get_future_state(self, gt_ego_fut_trajs, gt_ego_fut_masks, gt_agent_fut_trajs, gt_agent_fut_masks, gt_labels_3d):

        """get_future_label.
        Args:
            gt_ego_fut_trajs ([B, T1, 2]): ego future planning coordinate
            gt_ego_fut_masks ([B, T1]): ego future planning mask
            gt_agent_fut_trajs (B *[A, T2, 2]): agent future planning coordinate
            gt_agent_fut_masks (B *[A, T2]): agent future planning mask
        Returns:
            gt_trajs: [B, A+1, T * 2]
        """

        self.agent_indices = []
        # the number of agent anchor
        agent_dim = 900 
        # car, truck, bus, trailer, pedestrian
        veh_list = [0, 1, 3, 4, 8] 
        batch_size = len(gt_labels_3d)
        
        gt_fut_state_bs_list = []

        for bs in range(batch_size):
            gt_labels_3d_bs = gt_labels_3d[bs]
            gt_agent_fut_masks_bs = gt_agent_fut_masks[bs]
            gt_agent_fut_trajs_bs = gt_agent_fut_trajs[bs]
            gt_label_3d_mask_bs = torch.tensor([label in veh_list for label in gt_labels_3d_bs]).to(gt_agent_fut_masks_bs.device)
            self.agent_indices.append(torch.where(gt_label_3d_mask_bs)[0])
            if len(gt_label_3d_mask_bs) != 0:
                gt_agent_valid_trajs_bs = gt_agent_fut_trajs_bs[gt_label_3d_mask_bs][:, :6, :] # [A, T, 2]
            else:
                gt_agent_valid_trajs_bs = []

            if len(gt_agent_valid_trajs_bs) != 0 & len(gt_agent_valid_trajs_bs) < agent_dim:
                gt_fut_trajs = torch.cat(
                    (gt_agent_valid_trajs_bs,
                     torch.zeros([agent_dim - len(gt_agent_valid_trajs_bs), self.ego_fut_ts, 2], device=gt_agent_fut_masks_bs.device)), 0) # [N1, T, 2]
            else:
                gt_fut_trajs = torch.zeros([agent_dim, self.ego_fut_ts, 2], device=gt_agent_fut_masks_bs.device)

            gt_fut_state_bs_list.append(gt_fut_trajs)
        # driving demonstrations
        gt_trajs = torch.cat((torch.stack(gt_fut_state_bs_list), gt_ego_fut_trajs.unsqueeze(1).repeat(1, self.ego_fut_mode * 3, 1, 1)), dim=1)  # [B, N1 + M * 3, T, 2]

        return gt_trajs.flatten(-2, -1) # [B, N1 + ME, T * 2]


    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        """
        Args:
            present_features (B, N1, C): output features of transformer model.
            future_distribution_inputs (B, N1, T * 2): the agent and ego gt trajectory in the future.
            noise: gaussian noise.
        Returns:
            sample: sample tokens from present/future distribution
            present_distribution_mu: mean value of present gaussian distribution with shape (B, S, D)
            present_distribution_log_sigma: variance of present gaussian distribution with shape (B, S, D)
            future_distribution_mu: mean value of future gaussian distribution with shape (B, S, D)
            future_distribution_log_sigma: variance of future gaussian distribution with shape (B, S, D)
        """

        BS = present_features.shape[0]
        N1 = present_features.shape[1]
        # generative model for instance
        present_mu, present_log_sigma = self.present_distribution(present_features) # [B, N1, 32]

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            if self.future_concate is True:
                future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
            else:
                future_features = future_distribution_inputs
            # generative model for driving demonstrations
            future_mu, future_log_sigma = self.future_distribution(future_features) # [B, N1, 32]
         
        if noise is None:
            if self.training:
                noise = torch.randn_like(future_mu) # [B, N1, 32]
            else:
                noise = torch.randn_like(present_mu)

        if self.training:
            mu = future_mu
            sigma = torch.exp(self.log_paramter * future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(self.log_paramter * present_log_sigma)
        # distribution fusion
        sample = mu + sigma * noise # [B, N1, 32]

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.permute(0, 2, 1).expand(BS, self.latent_dim, N1) # [B, 32, N1]

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution


    def future_states_predict(self, sample, hidden_states, current_states):
        """
        Args:
            sample[B, 32, N1]: sample taken from present/future distribution
            hidden_states: hidden states input of autoregressive model.
            current_states: current states input of autoregressive model.
        Returns:
            states_hs: the final features combined with the generative features and current features
            future_states_hs: the generative features predicted by generate model(VAE)
        """

        BS, N1 = hidden_states.shape[:2]
        future_prediction_input = sample.unsqueeze(0).expand(self.temporal_frames, -1, -1, -1).permute(0, 1, 3, 2).contiguous() # [T, B, 32, N1]
        future_prediction_input = future_prediction_input.reshape(self.temporal_frames, -1, self.latent_dim)  # [T, B * N1, 32]
        hidden_state = hidden_states.clone().reshape(self.layer_dim, -1, self.embed_dims // self.layer_dim) # [L, B * N1, 64]
        # prediction model
        future_states = self.predict_model(future_prediction_input, hidden_state) # [T, B * N1, 256]

        if self.auto_regression:
            future_states = future_states.permute(1, 0, 2).reshape(BS, N1, -1, future_states.shape[2]) # [B, N1, T, C]
            current_states = current_states.unsqueeze(2).expand(-1, -1, self.temporal_frames, -1) # [B, N1, T, C]
        else:
            future_states = future_states.reshape(BS, N1, future_states.shape[2]) # [B, N1, 256]
        # residual blocks
        if self.with_cur:
            states_hs = current_states + future_states
        else:
            states_hs = future_states

        return states_hs, future_states

    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):

        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        # Calculate the gradient of the generative model
        if self.vae_model is True:
            distributioon_loss = self.loss_vae_gen(planning_model_outs['distribution'], motion_loss_cache, self.agent_indices)
            loss.update(distributioon_loss)    
        return loss

    def feature_aggregation(self,
        planning_model_outs,
        gt_ego_fut_cmd
    ):
        '''Feature alignment for knowledge distillation'''

        device = planning_model_outs['classification'][0].device
        B, M, T, D = planning_model_outs['prediction'][0].flatten(end_dim=1).shape
        cmd = gt_ego_fut_cmd.argmax(dim=-1)
        bs_indices = torch.arange(B, device=device)
        target = {}
        target['cls_weight'] = torch.ones((B), device=device).long()

        # knowledge distiilation select from command
        if self.select_command_kd:
            target['st_cls'] = planning_model_outs['classification'][0].flatten(end_dim=1).reshape(B, 3, self.ego_fut_mode)[bs_indices, cmd]
            target['tc_cls'] = planning_model_outs['teacher_classification'][0].flatten(end_dim=1).reshape(B, 3, self.ego_fut_mode)[bs_indices, cmd]
            target['st_reg'] = planning_model_outs['prediction'][0].flatten(end_dim=1).reshape(B, 3, self.ego_fut_mode, self.ego_fut_ts, 2)[bs_indices, cmd]
            target['tc_reg'] = planning_model_outs['teacher_prediction'][0].flatten(end_dim=1).reshape(B, 3, self.ego_fut_mode, self.ego_fut_ts, 2)[bs_indices, cmd]
            target['st_feat'] = planning_model_outs['feature'][0].flatten(end_dim=1).reshape(B, 3, self.ego_fut_mode, self.embed_dims)[bs_indices, cmd]
            target['tc_feat'] = planning_model_outs['teacher_feature'][0].flatten(end_dim=1).reshape(B, 3, self.ego_fut_mode, self.embed_dims)[bs_indices, cmd]
            target['st_en_feat'] = planning_model_outs['encoder_feature'].reshape(B, 3, self.ego_fut_mode, self.embed_dims)[bs_indices, cmd]
            target['tc_en_feat'] = planning_model_outs['teacher_encoder_feature'].reshape(B, 3, self.ego_fut_mode, self.embed_dims)[bs_indices, cmd]            
            target['reg_weight'] = torch.ones((B, M // 3, T, 1), device=device).long()
            target['feature_weight'] = torch.ones((B, M // 3, 1), device=device).long()
            target['feature_weight_total'] = torch.ones((B, M, 1), device=device).long()

        else:
            target['st_cls'] = planning_model_outs['classification'][0].flatten(end_dim=1)
            target['tc_cls'] = planning_model_outs['teacher_classification'][0].flatten(end_dim=1)
            target['st_reg'] = planning_model_outs['prediction'][0].flatten(end_dim=1)
            target['tc_reg'] = planning_model_outs['teacher_prediction'][0].flatten(end_dim=1)
            target['st_feat'] = planning_model_outs['feature'][0].flatten(end_dim=1)
            target['tc_feat'] = planning_model_outs['teacher_feature'][0].flatten(end_dim=1)
            target['st_en_feat'] = planning_model_outs['encoder_feature']
            target['tc_en_feat'] = planning_model_outs['teacher_encoder_feature']
            target['reg_weight'] = torch.ones((B, M, T, 1), device=device).long()
            target['feature_weight'] = torch.ones((B, M, 1), device=device).long()

        return target
    
    @force_fp32(apply_to=("model_outs"))
    def distillation_loss(self,
        planning_model_outs,
        gt_ego_fut_cmd,
    ):

        loss = {}
        target = self.feature_aggregation(planning_model_outs, gt_ego_fut_cmd)

        # Adpater for knownledge distillation
        if self.adapter_tag:
            target['st_feat'] = self.feature_adapter(target['st_feat'])

        loss['kd_loss_cls'] = self.kd_loss_cls(
            target['st_cls'],
            nn.Softmax(dim=1)(target['tc_cls']),
            weight=target['cls_weight'],
        )  

        loss['kd_loss_reg'] = self.kd_loss_reg(
            target['st_reg'],
            target['tc_reg'],
            weight=target['reg_weight'],
        )

        loss['kd_loss_feats'] = self.kd_loss_feats(
            target['st_feat'],
            target['tc_feat'],
            weight=target['feature_weight'], 
        )

        loss['kd_loss_en_feats'] = self.kd_loss_en_feats(
            target['st_en_feat'],
            target['tc_en_feat'],
            weight=target['feature_weight'],            
        )

        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)

            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])
            # reinforce learning loss
            if self.dqn_model is not None:
                planning_loss_rl = self.dqn_model(data['ego_status'], status.squeeze(1), reg_target, reg_pred, data['gt_ego_fut_cmd'])
            else:
                planning_loss_rl = torch.tensor(0.0, device=status_loss.device)

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                    f"planning_loss_rl_{decoder_idx}": planning_loss_rl,
                }
            )


        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"], 
            det_output["prediction"], 
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result