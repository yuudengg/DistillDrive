import torch

from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

# lateral order follows gt_ego_fut_cmd / plan_anchor groups: 0 right, 1 left, 2 straight
LATERAL_NAMES = ("right", "left", "straight")
LONGITUDINAL_NAMES = ("accelerate", "keep", "decelerate", "stop")
NUM_META_ACTIONS = len(LATERAL_NAMES) * len(LONGITUDINAL_NAMES) # 12


@PLUGIN_LAYERS.register_module()
class MetaActionHead(BaseModule):
    """Rule-based meta-action extraction from planned trajectories (no learnable params).

    meta_action = lateral * 4 + longitudinal, e.g. left(1) + decelerate(2) -> 6.
    The lateral rule is the same one used to build gt_ego_fut_cmd in
    tools/data_converter/nuscenes_converter.py (final x >= 2m right, <= -2m left).
    """

    def __init__(
        self,
        ego_fut_ts=6,
        ego_fut_mode=6,
        dt=0.5,
        lat_thresh=2.0,
        acc_ratio=1.2,
        dec_ratio=0.8,
        stop_speed=0.5,
        num_speed_steps=2,
    ):
        super(MetaActionHead, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.dt = dt # seconds between trajectory points
        self.lat_thresh = lat_thresh # m, final lateral offset to count as a turn
        self.acc_ratio = acc_ratio # late speed > early speed * acc_ratio -> accelerate
        self.dec_ratio = dec_ratio # late speed < early speed * dec_ratio -> decelerate
        self.stop_speed = stop_speed # m/s, late speed below this -> stop
        self.num_speed_steps = num_speed_steps # steps averaged for early / late speed

    @torch.no_grad()
    def forward(self, plan_reg, plan_cls=None, cmd=None):
        """
        Args:
            plan_reg: [B, 1, 3 * ME, T, 2], per-step offsets (as output by refine)
            plan_cls: [B, 1, 3 * ME], optional, used to pick the selected mode
            cmd: [B, 3] one-hot gt_ego_fut_cmd, optional, used to pick the selected mode
        Returns:
            dict with per-mode lateral / longitudinal / meta_action [B, 1, 3 * ME]
            and, if plan_cls and cmd are given, selected_mode / selected_meta_action [B]
        """
        plan_reg = plan_reg.detach()
        positions = plan_reg.cumsum(dim=-2) # [B, 1, M, T, 2]

        # =========== lateral: final x offset ===========
        final_x = positions[..., -1, 0] # [B, 1, M]
        lateral = torch.full_like(final_x, 2, dtype=torch.long) # straight
        lateral[final_x >= self.lat_thresh] = 0 # right
        lateral[final_x <= -self.lat_thresh] = 1 # left

        # =========== longitudinal: early vs late speed ===========
        speed = plan_reg.norm(dim=-1) / self.dt # [B, 1, M, T]
        k = self.num_speed_steps
        early_speed = speed[..., :k].mean(dim=-1) # [B, 1, M]
        late_speed = speed[..., -k:].mean(dim=-1) # [B, 1, M]
        longitudinal = torch.full_like(lateral, 1) # keep
        longitudinal[late_speed > early_speed * self.acc_ratio] = 0 # accelerate
        longitudinal[late_speed < early_speed * self.dec_ratio] = 2 # decelerate
        longitudinal[late_speed < self.stop_speed] = 3 # stop has priority

        meta_action = lateral * len(LONGITUDINAL_NAMES) + longitudinal # [B, 1, M]
        output = {
            "lateral": lateral,
            "longitudinal": longitudinal,
            "meta_action": meta_action,
        }

        # =========== selected mode: cmd group + best score (decoder.select without rescore) ===========
        if plan_cls is not None and cmd is not None:
            bs = plan_cls.shape[0]
            bs_indices = torch.arange(bs, device=plan_cls.device)
            cmd_idx = cmd.argmax(dim=-1) # [B]
            cls = plan_cls.detach().reshape(bs, 3, self.ego_fut_mode) # [B, 3, ME]
            mode_idx = cls[bs_indices, cmd_idx].argmax(dim=-1) # [B]
            selected_mode = cmd_idx * self.ego_fut_mode + mode_idx # [B]
            output["selected_mode"] = selected_mode
            output["selected_meta_action"] = meta_action[bs_indices, 0, selected_mode] # [B]
        return output

    @staticmethod
    def to_text(meta_action):
        """meta_action index (int) -> 'decelerate+left'"""
        meta_action = int(meta_action)
        lateral = meta_action // len(LONGITUDINAL_NAMES)
        longitudinal = meta_action % len(LONGITUDINAL_NAMES)
        return f"{LONGITUDINAL_NAMES[longitudinal]}+{LATERAL_NAMES[lateral]}"
