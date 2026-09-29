from typing import List, Optional, Tuple, Union
import warnings
import pickle
import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


@HEADS.register_module()
class DistillDriveHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        task_role,
        distillation_head = dict,
        det_head = dict,
        map_head = dict,
        motion_plan_head = dict,
        init_cfg=None,
        **kwargs,
    ):
        super(DistillDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        self.task_role = task_role
        
        if self.task_config['with_distillation']:
            self.distillation_head = build_head(distillation_head)
        if self.task_config['with_det']:
            self.det_head = build_head(det_head)
        if self.task_config['with_map']:
            self.map_head = build_head(map_head)
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_head(motion_plan_head)
        # Frozen Teacher Head
        if self.task_role == 'Student' and self.task_config['with_distillation']:
            for param in self.distillation_head.parameters():
                param.requires_grad = False

    def init_weights(self):
        if self.task_config['with_distillation']:
            self.distillation_head.init_weights()
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        run_teacher = self.task_config['with_distillation'] and (
            self.training or not self.task_config['with_motion_plan']
        )
        if run_teacher:
            motion_tc, planning_tc = self.distillation_head(metas)
        else:
            motion_output, planning_output = None, None
        if self.task_role == 'Teacher':
            motion_output, planning_output = motion_tc, planning_tc

        if self.task_config['with_det']:
            det_output = self.det_head(feature_maps, metas)
        else:
            det_output = None

        if self.task_config['with_map']:
            map_output = self.map_head(feature_maps, metas)
        else:
            map_output = None
        
        if self.task_config['with_motion_plan']:
            motion_output, planning_output = self.motion_plan_head(
                det_output, 
                map_output, 
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
            )

            if run_teacher:
                planning_output['teacher_classification'] = planning_tc['classification']
                planning_output['teacher_prediction'] = planning_tc['prediction']
                planning_output['teacher_encoder_feature'] = planning_tc['encoder_feature']
                planning_output['teacher_decoder_feature'] = planning_tc['decoder_feature']     
                planning_output['teacher_feature'] = planning_tc['feature']
        # TODO
        # analyse_dict = {}
        # analyse_dict['teacher_feature'] = planning_output['teacher_feature'][0][0][0].cpu()
        # analyse_dict['feature'] = planning_output['feature'][0][0][0].cpu()
        # with open('/iag_ad_01/ad/yurui/Project/SparseDrive/analyse/planning.pkl', 'wb') as file:
        #     pickle.dump(analyse_dict, file)

        return det_output, map_output, motion_output, planning_output

    def loss(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config['with_distillation'] and self.task_config['with_motion_plan'] is False:
            loss_distillation = self.distillation_head.loss(
                motion_output, 
                planning_output, 
                data, 
            )
            losses.update(loss_distillation)
        
        if self.task_config['with_det']:
            loss_det = self.det_head.loss(det_output, data)
            losses.update(loss_det)
        
        if self.task_config['with_map']:
            loss_map = self.map_head.loss(map_output, data)
            losses.update(loss_map)

        if self.task_config['with_motion_plan']:
            motion_loss_cache = dict(
                indices=self.det_head.sampler.indices, 
            )
            loss_motion = self.motion_plan_head.loss(
                motion_output, 
                planning_output, 
                data, 
                motion_loss_cache
            )
            losses.update(loss_motion)
            if self.task_config['with_distillation']:
                loss_kd = self.motion_plan_head.distillation_loss(planning_output, data['gt_ego_fut_cmd'])
                losses.update(loss_kd)
        
        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        if self.task_config['with_distillation'] and self.task_config['with_motion_plan'] is False:
            motion_result, planning_result = self.distillation_head.post_process(
                motion_output, 
                planning_output,
                data,
            )
            batch_size = len(motion_result)
        if self.task_config['with_det']:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)
        
        if self.task_config['with_map']:
            map_result= self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config['with_motion_plan']:
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output,
                motion_output, 
                planning_output,
                data,
            )

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_distillation'] and self.task_config['with_motion_plan'] is False:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map']:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results