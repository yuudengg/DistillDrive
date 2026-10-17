"""Connect C's language module to motion_planning_head.py (same change as language_packet_hook.patch).

Target: language branch after B's feedback fix (877c1f6). Run from the repository root:
    python tools/language/patches/apply_language_patch.py --check   # only check that every anchor is found
    python tools/language/patches/apply_language_patch.py           # edit in place (a .bak copy is written first)

B's feedback flow (per-sample trigger, infer_batch, None -> 1st pass, _select) is kept as is. Only C's part changes:
  1. import split_packet
  2. constructor: lang_packet_cfg, lang_packet_refine_idx, lang_packet_dump
  3. build the packet builder (required when language_module.enabled=True)
  4. feedback call: build the packet from the 1st-pass refine outputs (only when a sample is triggered) and query with it
  5. _scene_input = sample b of that packet (the TODO(C)); the text / embedding conversion happens inside C's module
  6. end of forward: packet in planning_output["lang_packet"] only while dumping (lang_packet_dump, off by default;
     attach_packet_dumper turns it on) -> no extra cost per frame otherwise

Why a script as well as the .patch: the file uses Windows line endings (CRLF), which makes `git apply` fragile when the
patch file is copied around. This script keeps the file's line ending and refuses to run if an anchor is missing.
"""
import argparse
import shutil
from pathlib import Path

TARGET = Path("projects/mmdet3d_plugin/models/motion/motion_planning_head.py")

EDITS = [
    (
        "from ..language import build_language_module\n",
        "from ..language import build_language_module\n"
        "from ..language.packet import split_packet\n",
    ),
    (
        "        num_det=50,\n        num_map=10,\n    ):\n",
        "        num_det=50,\n        num_map=10,\n        lang_packet_cfg=None,\n        lang_packet_refine_idx=-1,\n"
        "        lang_packet_dump=False,\n    ):\n",
    ),
    (
        "                \"language_module.enabled=True 인데 adaln/language_model 설정이 없습니다\"\n"
        "            )\n",
        "                \"language_module.enabled=True 인데 adaln/language_model 설정이 없습니다\"\n"
        "            )\n"
        "        # C: 언어모듈 입력(scene_input)을 만드는 패킷 빌더. 되먹임을 켜면 필수, 트리거가 걸린 때만 계산.\n"
        "        # lang_packet_dump: forward 끝 패킷을 planning_output에 저장 (덤프용, attach_packet_dumper가 켬)\n"
        "        # lang_packet_refine_idx: 그 덤프 패킷에 쓸 refine 출력 (-1 = 마지막)\n"
        "        self.lang_packet_builder = build(lang_packet_cfg, PLUGIN_LAYERS)\n"
        "        self.lang_packet_refine_idx = lang_packet_refine_idx\n"
        "        self.lang_packet_dump = lang_packet_dump\n"
        "        if self.use_feedback:\n"
        "            assert self.lang_packet_builder is not None, \"language feedback needs lang_packet_cfg (scene_input comes from the packet)\"\n",
    ),
    (
        "                        meta_idx, use = self._query_language(instance_feature, metas, use)\n",
        "                        packet = self._lang_packet(\n"
        "                            instance_feature, num_anchor, det_confidence, det_anchors, det_classification,\n"
        "                            map_instance_feature_selected, plan_query, plan_reg, plan_cls, metas,\n"
        "                        )\n"
        "                        meta_idx, use = self._query_language(packet, use)\n",
    ),
    (
        "    def _query_language(self, instance_feature, metas, use):\n",
        "    def _query_language(self, packet, use):\n",
    ),
    (
        "            [self._scene_input(instance_feature, metas, b) for b in idx]\n",
        "            [self._scene_input(packet, b) for b in idx]\n",
    ),
    (
        "    def _scene_input(self, instance_feature, metas, b):\n"
        "        \"\"\"언어모듈 입력 (샘플 b 하나). 팀 합의: 1차는 '텍스트 요약' 형식.\n"
        "        TODO(C): scene_input 최종 형식(텍스트 요약) 확정 후 여기 맞추기.\"\"\"\n"
        "        bs = instance_feature.shape[0]\n"
        "\n"
        "        def pick(v):\n"
        "            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == bs:\n"
        "                return v[b]\n"
        "            if isinstance(v, (list, tuple)) and len(v) == bs:\n"
        "                return v[b]\n"
        "            return v\n"
        "\n"
        "        if isinstance(metas, dict):\n"
        "            metas = {k: pick(v) for k, v in metas.items()}\n"
        "        return {\"instance_feature\": instance_feature[b], \"metas\": metas}\n",
        "    def _lang_packet(self, instance_feature, num_anchor, det_confidence, det_anchors, det_classification,\n"
        "                     map_feat, plan_query, plan_reg, plan_cls, metas):\n"
        "        \"\"\"C: 1차 refine 결과로 언어 패킷 (배치 전체, detach). 트리거가 걸린 때만 호출된다.\"\"\"\n"
        "        return self.lang_packet_builder(\n"
        "            agent_feat=instance_feature[:, :num_anchor],\n"
        "            agent_conf=det_confidence,\n"
        "            agent_boxes=det_anchors,\n"
        "            agent_labels=det_classification.argmax(dim=-1),\n"
        "            ego_feat_modes=plan_query[:, 0],  # [B, 1, 3 * ME, D] -> [B, 3 * ME, D]\n"
        "            plan_reg=plan_reg,\n"
        "            plan_cls=plan_cls,\n"
        "            cmd=metas.get(\"gt_ego_fut_cmd\") if isinstance(metas, dict) else None,\n"
        "            map_feat=map_feat,\n"
        "        )\n"
        "\n"
        "    @staticmethod\n"
        "    def _scene_input(packet, b):\n"
        "        \"\"\"언어모듈 입력 (샘플 b 하나) = 패킷의 b번째. 텍스트/임베딩 경로 모두 같은 dict이고 변환은 C 안에서 한다.\"\"\"\n"
        "        return split_packet(packet, b)\n",
    ),
    (
        "        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)\n",
        "        # =========== language packet for dumps (eval only, detached, does not change any driving output) ===========\n"
        "        lang_packet = None\n"
        "        if self.lang_packet_dump and self.lang_packet_builder is not None and not self.training:\n"
        "            ref = self.lang_packet_refine_idx\n"
        "            lang_packet = self.lang_packet_builder(\n"
        "                agent_feat=instance_feature[:, :num_anchor],\n"
        "                agent_conf=det_confidence,\n"
        "                agent_boxes=det_anchors,\n"
        "                agent_labels=det_classification.argmax(dim=-1),\n"
        "                ego_feat_modes=planning_feature[ref][:, 0],  # plan_query [B, 1, 3 * ME, D] -> [B, 3 * ME, D]\n"
        "                plan_reg=planning_prediction[ref],\n"
        "                plan_cls=planning_classification[ref],\n"
        "                cmd=metas.get(\"gt_ego_fut_cmd\") if isinstance(metas, dict) else None,\n"
        "                map_feat=map_instance_feature_selected,\n"
        "            )\n"
        "\n"
        "        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)\n",
    ),
    (
        "        # Add distribution to calculate the gradient\n",
        "        if lang_packet is not None:\n"
        "            planning_output[\"lang_packet\"] = lang_packet\n"
        "        # Add distribution to calculate the gradient\n",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(TARGET))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    raw = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    if "lang_packet_builder" in raw:
        raise SystemExit("already patched (lang_packet_builder found) - nothing to do")

    text = raw
    for i, (old, new) in enumerate(EDITS, 1):
        old_nl, new_nl = old.replace("\n", nl), new.replace("\n", nl)
        count = text.count(old_nl)
        if count != 1:
            raise SystemExit(f"edit {i}: anchor found {count} times (expected 1) - the file differs, merge by hand:\n{old}")
        text = text.replace(old_nl, new_nl)
    print(f"all {len(EDITS)} anchors found (line ending: {'CRLF' if nl == chr(13) + chr(10) else 'LF'})")
    if args.check:
        return
    shutil.copyfile(path, str(path) + ".bak")
    path.write_bytes(text.encode("utf-8"))
    print(f"patched {path}  (backup: {path}.bak)")


if __name__ == "__main__":
    main()
