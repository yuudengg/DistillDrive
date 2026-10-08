"""Offline "reason" label generation with a vision-language model (default Qwen2.5-VL).  NEEDS A GPU.

    python tools/language/gen_reason_labels.py \
        --pkl data/infos/nuscenes_infos_train.pkl \
        --meta-json data/language/meta_action_gt_train.json \
        --data-root data/nuscenes \
        --out data/language/reasons_train.jsonl --stride 4 --batch-size 8

Design points
  * the VLM is a SEPARATE model from the language model that is trained later (label model != student LM)
  * the VLM sees the front camera image + the GT meta-action and writes ONE grounded sentence
  * resumable: rows already in --out are skipped; rejected outputs go to <out>.rejected.jsonl
  * --stride N takes every N-th sample, handy for a quick first pass before spending GPU hours on all of train

CPU check of the whole pipeline (pkl -> prompt -> clean/validate -> jsonl) without a VLM:  add  --mock
(images are not opened, the "VLM" returns a template sentence for the meta-action)

Not verified on a real GPU here (the writer's laptop has none): run with --limit 16 first and read the output.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import load_language_package  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--meta-json", required=True, help="output of make_meta_action_gt.py")
    ap.add_argument("--data-root", default="", help="prefix for relative image paths in the pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="Qwen/Qwen2.5-VL-3B-Instruct or Qwen/Qwen2.5-VL-7B-Instruct (better labels, ~16GB+ GPU)")
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--max-words", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=640 * 28 * 28)
    ap.add_argument("--mock", action="store_true", help="no VLM / no images: template sentences (CPU pipeline test)")
    args = ap.parse_args()

    import pickle

    lang = load_language_package()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) else data
    with open(args.meta_json) as f:
        meta_gt = json.load(f)

    todo = []
    for info in infos[:: args.stride]:
        tok = info["token"]
        if tok not in meta_gt:
            continue
        path = info["cams"][args.cam]["data_path"]
        if args.data_root and not os.path.isabs(path):
            path = os.path.join(args.data_root, path)
        todo.append((tok, path, int(meta_gt[tok])))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["sample_token"] for l in f}
    todo = [t for t in todo if t[0] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[info] to label: {len(todo)}  (already done: {len(done)})")

    if args.mock:
        args.model = "mock"

        def generate(chunk):
            return [f"Reason: I will {lang.action_phrase(meta, 'en')} because the lane ahead allows it." for _, _, meta in chunk]
    else:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        processor.tokenizer.padding_side = "left"

        def generate(chunk):
            texts, images = [], []
            for tok, path, meta in chunk:
                prompt = lang.build_label_prompt(meta, args.lang, args.max_words)
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                images.append(Image.open(path).convert("RGB"))
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            return processor.batch_decode(out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_ok = n_rej = 0
    with open(args.out, "a", encoding="utf-8") as fout, open(args.out + ".rejected.jsonl", "a", encoding="utf-8") as frej:
        for s in range(0, len(todo), args.batch_size):
            chunk = todo[s : s + args.batch_size]
            gen = generate(chunk)
            for (tok, _, meta), raw in zip(chunk, gen):
                reason = lang.clean_reason(raw)
                row = {
                    "sample_token": tok,
                    "meta_action": meta,
                    "meta_action_text": lang.action_tag(meta),
                    "reason": reason,
                    "label_model": args.model,
                }
                if lang.validate_reason(reason):
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_ok += 1
                else:
                    frej.write(json.dumps(dict(row, raw=raw), ensure_ascii=False) + "\n")
                    n_rej += 1
            fout.flush()
            frej.flush()
            if (s // args.batch_size) % 20 == 0:
                print(f"[{s + len(chunk)}/{len(todo)}] ok {n_ok}  rejected {n_rej}", flush=True)
    print(f"[done] ok {n_ok}  rejected {n_rej}  -> {args.out}")


if __name__ == "__main__":
    main()
