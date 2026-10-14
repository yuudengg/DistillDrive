"""Language packet = the small, detached bundle the language branch needs from the driving model.

Why a packet instead of calling the LM inside MotionPlanningHead.forward:
  * the 3B LM never touches the driving model's training / eval  -> existing results stay identical
  * the GPU person dumps packets ONCE (tools: attach_packet_dumper); the language team then trains on the
    cached packets and can debug on CPU with a tiny LM
  * the packet is ~20 tokens x 256 dims per sample, so caching a whole split is a few hundred MB

Layout of packet["scene"] along dim 1:   [ ego (1) | agents (K) | map (M) ]
"""
import os
from typing import Dict, List

import torch
import torch.nn as nn

try:  # registry only exists in the real repo env; CPU test env has no mmcv
    from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

    _register = PLUGIN_LAYERS.register_module()
except Exception:  # pragma: no cover

    def _register(cls):
        return cls


TYPE_EGO, TYPE_AGENT, TYPE_MAP = 0, 1, 2


def _as_cmd(cmd, bs, device):
    """gt_ego_fut_cmd may arrive as tensor [B,3] or a list of [3] tensors / arrays."""
    if cmd is None:
        return None
    if not torch.is_tensor(cmd):
        cmd = torch.stack([torch.as_tensor(c) for c in cmd])
    cmd = cmd.to(device).float()
    if cmd.dim() == 1:
        cmd = cmd[None]
    assert cmd.shape == (bs, 3), f"cmd shape {tuple(cmd.shape)} != ({bs}, 3)"
    return cmd


@_register
class LangPacketBuilder(nn.Module):
    """Builds the language packet from tensors that already exist in MotionPlanningHead.forward.

    No learnable parameters. Everything is detached.
    """

    def __init__(
        self,
        num_agents=12,
        num_map=6,
        ego_fut_ts=6,
        ego_fut_mode=6,
        agent_conf_thresh=0.2,
        meta_action=None,  # kwargs for MetaActionHead (dt, lat_thresh, ...)
        meta_action_head=None,  # inject an instance (tests); default is built lazily
        store_dtype="float16",
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_map = num_map
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.agent_conf_thresh = agent_conf_thresh
        self.store_dtype = getattr(torch, store_dtype)
        self._meta_cfg = dict(meta_action or {})
        self._meta_head = meta_action_head

    @property
    def meta_head(self):
        if self._meta_head is None:
            # lazy import: avoids a circular import (motion_planning_head imports this module)
            from ..motion.meta_action import MetaActionHead

            self._meta_head = MetaActionHead(
                ego_fut_ts=self.ego_fut_ts, ego_fut_mode=self.ego_fut_mode, **self._meta_cfg
            )
        return self._meta_head

    @property
    def num_tokens(self):
        return 1 + self.num_agents + self.num_map

    @torch.no_grad()
    def forward(
        self,
        agent_feat,  # [B, N, D]   agent instance features (after the refine stack)
        agent_conf,  # [B, N]      detection confidence (sigmoid max)
        agent_boxes,  # [B, N, >=2] boxes, first two dims = x, y in ego frame
        agent_labels,  # [B, N]      class ids
        ego_feat_modes,  # [B, 3*ME, D] per-mode ego query (planning_feature[idx][:, 0])
        plan_reg,  # [B, 1, 3*ME, T, 2] per-step offsets (as output by refine)
        plan_cls,  # [B, 1, 3*ME]
        cmd=None,  # [B, 3] gt_ego_fut_cmd; if None the globally best mode is used
        map_feat=None,  # [B, M', D]
    ) -> Dict[str, torch.Tensor]:
        bs, n_agent, dim = agent_feat.shape
        device = agent_feat.device
        b_idx = torch.arange(bs, device=device)

        # ---------- meta-action of the selected mode (rule based, from the planned trajectory) ----------
        cmd = _as_cmd(cmd, bs, device)
        out = self.meta_head(plan_reg, plan_cls if cmd is not None else None, cmd)
        if cmd is not None:
            sel = out["selected_mode"]
        else:
            sel = plan_cls.detach().reshape(bs, -1).argmax(dim=-1)
        meta = out["meta_action"][b_idx, 0, sel]  # [B]

        # ---------- selected trajectory as positions ----------
        traj = plan_reg.detach().cumsum(dim=-2)[b_idx, 0, sel]  # [B, T, 2]

        # ---------- ego token ----------
        ego_tok = ego_feat_modes.detach()[b_idx, sel]  # [B, D]

        # ---------- agent tokens: top-K by confidence ----------
        k = min(self.num_agents, n_agent)
        top_conf, top_idx = agent_conf.detach().topk(k, dim=1)  # [B, K]
        feat = torch.gather(agent_feat.detach(), 1, top_idx[..., None].expand(-1, -1, dim))
        xy = torch.gather(agent_boxes.detach()[..., :2], 1, top_idx[..., None].expand(-1, -1, 2))
        lab = torch.gather(agent_labels.detach(), 1, top_idx)
        agent_valid = top_conf >= self.agent_conf_thresh
        pad = self.num_agents - k
        if pad > 0:
            feat = torch.cat([feat, feat.new_zeros(bs, pad, dim)], 1)
            xy = torch.cat([xy, xy.new_zeros(bs, pad, 2)], 1)
            lab = torch.cat([lab, lab.new_zeros(bs, pad)], 1)
            top_conf = torch.cat([top_conf, top_conf.new_zeros(bs, pad)], 1)
            agent_valid = torch.cat([agent_valid, agent_valid.new_zeros(bs, pad)], 1)

        # ---------- map tokens ----------
        if map_feat is None:
            map_feat = agent_feat.new_zeros(bs, 0, dim)
        m = min(self.num_map, map_feat.shape[1])
        maps = map_feat.detach()[:, :m]
        map_valid = torch.ones(bs, m, dtype=torch.bool, device=device)
        if self.num_map - m > 0:
            maps = torch.cat([maps, maps.new_zeros(bs, self.num_map - m, dim)], 1)
            map_valid = torch.cat([map_valid, map_valid.new_zeros(bs, self.num_map - m)], 1)

        scene = torch.cat([ego_tok[:, None], feat, maps], dim=1)  # [B, 1+K+M, D]
        scene_mask = torch.cat([torch.ones(bs, 1, dtype=torch.bool, device=device), agent_valid, map_valid], 1)
        scene_type = torch.cat(
            [
                torch.full((1,), TYPE_EGO, dtype=torch.long),
                torch.full((self.num_agents,), TYPE_AGENT, dtype=torch.long),
                torch.full((self.num_map,), TYPE_MAP, dtype=torch.long),
            ]
        ).to(device)[None].expand(bs, -1)

        return {
            "scene": scene.to(self.store_dtype),
            "scene_mask": scene_mask,
            "scene_type": scene_type.contiguous(),
            "traj": traj.float(),  # [B, T, 2] positions
            "meta_action": meta.long(),  # [B] 0..11 predicted by the 1st-pass trajectory
            "selected_mode": sel.long(),  # [B]
            "agent_xy": xy.float(),  # [B, K, 2]
            "agent_labels": lab.long(),  # [B, K]
            "agent_conf": top_conf.float(),  # [B, K]
        }


# ----------------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------------
def split_packet(packet: Dict[str, torch.Tensor], i: int) -> Dict[str, torch.Tensor]:
    return {k: v[i].detach().cpu().clone() for k, v in packet.items()}


def save_packets(packet: Dict[str, torch.Tensor], sample_ids: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, sid in enumerate(sample_ids):
        torch.save(split_packet(packet, i), os.path.join(out_dir, f"{sid}.pt"))


def load_packet(path: str) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # old torch without weights_only
        return torch.load(path, map_location="cpu")


def _unwrap(x):
    """mmcv DataContainer / nested lists -> flat list of dicts."""
    if hasattr(x, "data"):
        x = x.data
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]
    return x


def extract_sample_ids(metas, bs: int, counter: int) -> List[str]:
    """Best-effort sample token lookup; falls back to a running counter. CHECK the first dumped file names."""
    ids = None
    if isinstance(metas, dict):
        for key in ("sample_idx", "token", "sample_token"):
            if key in metas:
                ids = _unwrap(metas[key])
                break
        if ids is None and "img_metas" in metas:
            im = _unwrap(metas["img_metas"])
            if isinstance(im, (list, tuple)) and len(im) == bs and isinstance(im[0], dict):
                for key in ("sample_idx", "token", "sample_token"):
                    if key in im[0]:
                        ids = [m[key] for m in im]
                        break
    if ids is not None and not isinstance(ids, (list, tuple)):
        ids = [ids]
    if ids is None or len(ids) != bs:
        return [f"idx{counter + i:07d}" for i in range(bs)]
    return [str(i) for i in ids]


class PacketDumper:
    """Wraps MotionPlanningHead.forward: writes planning_output['lang_packet'] to <out_dir>/<token>.pt.

    Wrapping forward (instead of register_forward_hook(with_kwargs=True), torch >= 2.0 only) works on torch 1.x too.
    """

    def __init__(self, out_dir: str, overwrite: bool = False):
        self.out_dir = out_dir
        self.overwrite = overwrite
        self.counter = 0
        self.num_saved = 0
        self.head = None

    def _hook(self, module, args, kwargs, output):
        planning_output = output[1]
        packet = planning_output.get("lang_packet") if isinstance(planning_output, dict) else None
        if packet is None:
            return
        bs = packet["scene"].shape[0]
        metas = args[3] if len(args) > 3 else kwargs.get("metas")
        ids = extract_sample_ids(metas, bs, self.counter)
        self.counter += bs
        if not self.overwrite:
            keep = [i for i, sid in enumerate(ids) if not os.path.exists(os.path.join(self.out_dir, f"{sid}.pt"))]
            if not keep:
                return
            packet = {k: v[keep] for k, v in packet.items()}
            ids = [ids[i] for i in keep]
        save_packets(packet, ids, self.out_dir)
        self.num_saved += len(ids)

    def attach(self, model: nn.Module):
        heads = [m for m in model.modules() if m.__class__.__name__ == "MotionPlanningHead"]
        if not heads:
            raise RuntimeError("MotionPlanningHead not found in model")
        head = heads[0]
        if getattr(head, "lang_packet_builder", None) is None:
            raise RuntimeError("head.lang_packet_builder is None: set lang_packet_cfg in the config")
        orig = head.forward  # bound method; nn.Module.__call__ picks up the instance attribute below

        def forward(*args, **kwargs):
            output = orig(*args, **kwargs)
            self._hook(head, args, kwargs, output)
            return output

        head.forward = forward
        self.head = head
        return self

    def remove(self):
        if self.head is not None:
            del self.head.forward  # back to the class method
            self.head = None


def attach_packet_dumper(model: nn.Module, out_dir: str, overwrite: bool = False) -> PacketDumper:
    """One-liner for tools/test.py: call right after the checkpoint is loaded, then run the normal test loop."""
    return PacketDumper(out_dir, overwrite).attach(model)
