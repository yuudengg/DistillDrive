"""Synthetic planning trajectories for testing MetaActionHead without a GPU / real model.

Output follows the refine output plan_reg: per-step offsets (x, y), x > 0 = right,
groups ordered like gt_ego_fut_cmd (0 right, 1 left, 2 straight).
"""
import math

import torch

LATERAL_NAMES = ("right", "left", "straight")
LONGITUDINAL_NAMES = ("accelerate", "keep", "decelerate", "stop")

# forward speed (m/s) per future step, chosen with margin from MetaActionHead thresholds
SPEED_PROFILES = {
    "accelerate": [2.0, 2.0, 3.0, 4.0, 6.0, 6.0],
    "keep": [5.0] * 6,
    "decelerate": [6.0, 6.0, 5.0, 4.0, 3.0, 3.0],
    "stop": [5.0, 4.0, 3.0, 1.5, 0.0, 0.0], # turn is finished before the car stops
}
LAT_OFFSET = 3.0 # m, final lateral offset for a turn (rule threshold is 2m)


def make_single_traj(lateral, longitudinal, dt=0.5, speed_scale=1.0, lat_offset=LAT_OFFSET):
    """One trajectory of the given meta-action.

    Args:
        lateral: 0 right, 1 left, 2 straight
        longitudinal: 0 accelerate, 1 keep, 2 decelerate, 3 stop
    Returns:
        offsets [T, 2]; speed of every step equals the profile speed
    """
    speed = torch.tensor(SPEED_PROFILES[LONGITUDINAL_NAMES[longitudinal]]) * speed_scale
    step_len = speed * dt
    heading = 0.0
    if lateral != 2:
        # constant heading so that the final x is exactly +-lat_offset
        assert step_len.sum() > lat_offset, "trajectory too short for this lateral offset"
        heading = math.asin(lat_offset / float(step_len.sum()))
        if lateral == 1:
            heading = -heading
    return torch.stack([step_len * math.sin(heading), step_len * math.cos(heading)], dim=-1)


def make_batch(batch_size=1, ego_fut_mode=6, dt=0.5, seed=0):
    """plan_reg-shaped synthetic dump.

    Mode group g holds turns of lateral g (like plan_anchor groups); longitudinal is random.
    Returns:
        plan_reg [B, 1, 3 * ME, T, 2], meta_action [B, 1, 3 * ME] (expected labels)
    """
    gen = torch.Generator().manual_seed(seed)
    trajs, labels = [], []
    for _ in range(batch_size):
        for lateral in range(3):
            for _ in range(ego_fut_mode):
                longitudinal = int(torch.randint(4, (1,), generator=gen))
                scale = float(torch.empty(1).uniform_(0.9, 1.1, generator=gen))
                lat_offset = float(torch.empty(1).uniform_(2.5, 4.0, generator=gen))
                trajs.append(make_single_traj(lateral, longitudinal, dt, scale, lat_offset))
                labels.append(lateral * len(LONGITUDINAL_NAMES) + longitudinal)
    T = trajs[0].shape[0]
    plan_reg = torch.stack(trajs).reshape(batch_size, 1, 3 * ego_fut_mode, T, 2)
    meta_action = torch.tensor(labels).reshape(batch_size, 1, 3 * ego_fut_mode)
    return plan_reg, meta_action
