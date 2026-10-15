import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .packet import load_packet
from .prompt import describe_objects, scene_text_from_info, validate_reason


class ReasonPacketDataset(Dataset):
    """label jsonl rows  {"sample_token", "meta_action", "reason", ...}  +  <packet_dir>/<sample_token>.pt

    meta_source: which meta-action goes into the prompt
        "gt"   : rule applied to the GT trajectory (what the reason was written for)  -> default for training
        "pred" : rule applied to the model's own 1st-pass trajectory                   -> what inference will see
    with_objects: add 'Nearby objects: ...' text from detected agents (ablation lever)
    """

    def __init__(
        self,
        label_file: str,
        packet_dir: str,
        meta_source: str = "gt",
        with_objects: bool = False,
        lang: str = "en",
        object_topn: int = 5,
        validate: bool = True,
        max_samples: int = None,
    ):
        assert meta_source in ("gt", "pred")
        self.packet_dir, self.meta_source = packet_dir, meta_source
        self.with_objects, self.lang, self.object_topn = with_objects, lang, object_topn
        rows, n_missing, n_bad = [], 0, 0
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if not os.path.exists(os.path.join(packet_dir, f"{row['sample_token']}.pt")):
                    n_missing += 1
                    continue
                if validate and not validate_reason(row["reason"]):
                    n_bad += 1
                    continue
                rows.append(row)
        if max_samples:
            rows = rows[:max_samples]
        self.rows = rows
        self.stats = dict(kept=len(rows), missing_packet=n_missing, invalid_reason=n_bad)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Dict:
        row = self.rows[idx]
        pkt = load_packet(os.path.join(self.packet_dir, f"{row['sample_token']}.pt"))
        pred = int(pkt["meta_action"])
        gt = int(row["meta_action"])
        objects = None
        if self.with_objects:
            objects = describe_objects(pkt["agent_xy"], pkt["agent_labels"], pkt["agent_conf"], topn=self.object_topn)
        return {
            "sample_token": row["sample_token"],
            "scene": pkt["scene"].float(),
            "scene_mask": pkt["scene_mask"],
            "scene_type": pkt["scene_type"],
            "traj": pkt["traj"].float(),
            "meta_action": gt if self.meta_source == "gt" else pred,
            "gt_meta_action": gt,
            "pred_meta_action": pred,
            "objects": objects,
            "reason": row["reason"],
        }


class ReasonTextDataset(Dataset):
    """Text route: label jsonl rows + scene sentences built from the nuScenes info pkl (GT boxes, GT ego future).

    No packets / driving model needed, so the LM can be trained before the packet hook is in place.
    pred_meta_action == gt here (there is no 1st-pass trajectory in this source).
    """

    def __init__(self, label_file: str, info_pkl: str, validate: bool = True, max_samples: int = None, object_topn: int = 8):
        import pickle

        with open(info_pkl, "rb") as f:
            data = pickle.load(f)
        infos = data["infos"] if isinstance(data, dict) else data
        by_token = {info["token"]: info for info in infos}
        rows, texts, n_missing, n_bad = [], {}, 0, 0
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                info = by_token.get(row["sample_token"])
                if info is None:
                    n_missing += 1
                    continue
                if validate and not validate_reason(row["reason"]):
                    n_bad += 1
                    continue
                texts[row["sample_token"]] = scene_text_from_info(info, topn=object_topn)
                rows.append(row)
        if max_samples:
            rows = rows[:max_samples]
        self.rows, self.texts = rows, texts
        self.stats = dict(kept=len(rows), missing_info=n_missing, invalid_reason=n_bad)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Dict:
        row = self.rows[idx]
        meta = int(row["meta_action"])
        return {
            "sample_token": row["sample_token"],
            "text": self.texts[row["sample_token"]],
            "meta_action": meta,
            "gt_meta_action": meta,
            "pred_meta_action": meta,
            "objects": None,
            "reason": row["reason"],
        }


def collate_fn(items: List[Dict]) -> Dict:
    batch = {k: torch.stack([it[k] for it in items]) for k in ("scene", "scene_mask", "scene_type", "traj") if k in items[0]}
    for k in ("meta_action", "gt_meta_action", "pred_meta_action"):
        batch[k] = torch.tensor([it[k] for it in items], dtype=torch.long)
    for k in ("sample_token", "objects", "reason", "text"):
        if k in items[0]:
            batch[k] = [it[k] for it in items]
    return batch


def move_batch(batch: Dict, device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
