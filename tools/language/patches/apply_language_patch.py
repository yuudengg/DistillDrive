"""Connect C's language module to A's motion_planning_head.py (same change as language_packet_hook.patch).

Target: A's feedback integration (language branch a8930b8). Run from the repository root:
    python tools/language/patches/apply_language_patch.py --check   # only check that every anchor is found
    python tools/language/patches/apply_language_patch.py           # edit in place (a .bak copy is written first)

What changes (motion_planning_head.py is A's file, so A applies this):
  1. import LangPacketBuilder / split_packet / feedback helpers (the import also registers LanguageModule in PLUGIN_LAYERS)
  2. constructor: lang_packet_cfg, lang_packet_refine_idx
  3. build the packet builder (required when language_module.enabled=True)
  4. feedback call: per-sample infer_batch -> [B] meta -> adaln -> 2nd refine; failed samples keep the 1st pass
  5. _scene_input: packet from the 1st-pass refine outputs, one dict per sample (A's TODO(C))
  6. end of forward: optional detached packet in planning_output["lang_packet"] (eval only, for dumps)

Why a script as well as the .patch: the file uses Windows line endings (CRLF), which makes `git apply` fragile when the
patch file is copied around. This script keeps the file's line ending and refuses to run if an anchor is missing.
"""
import argparse
import shutil
from pathlib import Path

TARGET = Path("projects/mmdet3d_plugin/models/motion/motion_planning_head.py")

EDITS = [
    (
        "from ..instance_bank import topk\n",
        "from ..instance_bank import topk\n"
        "from ..language.packet import LangPacketBuilder, split_packet  # noqa: F401  (also registers LanguageModule in PLUGIN_LAYERS)\n"
        "from ..language.language_module import feedback_meta, select_per_sample\n",
    ),
    (
        "        num_det=50,\n        num_map=10,\n    ):\n",
        "        num_det=50,\n        num_map=10,\n        lang_packet_cfg=None,\n        lang_packet_refine_idx=-1,\n    ):\n",
    ),
    (
        "                \"language_module.enabled=True 인데 adaln/language_model 설정이 없습니다\"\n"
        "            )\n",
        "                \"language_module.enabled=True 인데 adaln/language_model 설정이 없습니다\"\n"
        "            )\n"
        "        # language packet (C): scene_input for the feedback call + optional dump. None = off.\n"
        "        # lang_packet_refine_idx: which refine output the end-of-forward packet uses (-1 = last)\n"
        "        self.lang_packet_builder = build(lang_packet_cfg, PLUGIN_LAYERS)\n"
        "        self.lang_packet_refine_idx = lang_packet_refine_idx\n"
        "        if self.use_feedback:\n"
        "            assert self.lang_packet_builder is not None, \"language feedback needs lang_packet_cfg (scene_input comes from the packet)\"\n",
    ),
    (
        "                if self.use_feedback and self._trigger(plan_cls):\n"
        "                    meta_idx, _sentence = self.language_model.infer(\n"
        "                        self._scene_input(instance_feature, metas)\n"
        "                    )\n"
        "                    cond = self.adaln(meta_idx)\n"
        "                    (\n"
        "                        motion_cls,\n"
        "                        motion_reg,\n"
        "                        plan_cls,\n"
        "                        plan_reg,\n"
        "                        plan_status,\n"
        "                        plan_query,\n"
        "                    ) = self._refine_once(\n",
        "                if self.use_feedback and self._trigger(plan_cls):\n"
        "                    # C: per-sample scene_input -> infer_batch; samples whose LM call failed keep the 1st pass\n"
        "                    results = self.language_model.infer_batch(\n"
        "                        self._scene_input(\n"
        "                            instance_feature, num_anchor, det_confidence, det_anchors, det_classification,\n"
        "                            map_instance_feature_selected, plan_query, plan_reg, plan_cls, metas,\n"
        "                        )\n"
        "                    )\n"
        "                    ok, meta_idx = feedback_meta(results, device=plan_cls.device)\n"
        "                    cond = self.adaln(meta_idx)  # [B] meta-action -> (gamma, beta, alpha)\n"
        "                    first = (motion_cls, motion_reg, plan_cls, plan_reg, plan_status, plan_query)\n"
        "                    second = self._refine_once(\n",
    ),
    (
        "                        cond=cond,\n"
        "                    )\n"
        "                motion_classification.append(motion_cls)\n",
        "                        cond=cond,\n"
        "                    )\n"
        "                    (\n"
        "                        motion_cls,\n"
        "                        motion_reg,\n"
        "                        plan_cls,\n"
        "                        plan_reg,\n"
        "                        plan_status,\n"
        "                        plan_query,\n"
        "                    ) = select_per_sample(ok, first, second)\n"
        "                motion_classification.append(motion_cls)\n",
    ),
    (
        "    def _scene_input(self, instance_feature, metas):\n"
        "        \"\"\"언어모듈 입력. 팀 합의: 1차는 '텍스트 요약' 형식.\n"
        "        TODO(C): scene_input 최종 형식(텍스트 요약) 확정 후 여기 맞추기.\"\"\"\n"
        "        return {\"instance_feature\": instance_feature, \"metas\": metas}\n",
        "    def _scene_input(self, instance_feature, num_anchor, det_confidence, det_anchors, det_classification,\n"
        "                     map_feat, plan_query, plan_reg, plan_cls, metas):\n"
        "        \"\"\"언어모듈 입력 (C): 1차 refine 결과로 패킷을 만들어 샘플별 dict 리스트로 돌려준다.\n"
        "        텍스트 경로든 임베딩 경로든 같은 dict이고, 문장 변환은 언어모듈 안에서 한다.\"\"\"\n"
        "        packet = self.lang_packet_builder(\n"
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
        "        return [split_packet(packet, b) for b in range(plan_reg.shape[0])]\n",
    ),
    (
        "        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)\n",
        "        # =========== language packet (eval only, detached, does not change any driving output) ===========\n"
        "        lang_packet = None\n"
        "        if self.lang_packet_builder is not None and not self.training:\n"
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
