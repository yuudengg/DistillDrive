"""Evaluate the language head on a validation split.

    python tools/language/eval_language.py --ckpt work_dirs/language/stage2/best --stage 2 \
        --lm Qwen/Qwen1.5-1.8B-Chat --load-8bit --labels data/language/reasons_val.jsonl --packets data/language/packets_val \
        --meta-source pred --out work_dirs/language/stage2/eval_pred.jsonl

Reported
  rougeL        : word-level ROUGE-L F1 against the VLM label (weak proxy: wording can differ and still be right)
  consistency   : does the generated reason mention the action (turn left / slow down ...)?   0..1
  valid_rate    : fraction passing validate_reason (length, no 'the image', ...)
  meta acc      : accuracy of the 1st-pass trajectory's meta-action vs the GT meta-action
                  (lateral-only and full 12-class)  -> tells you how good the thing the language is conditioned on is
  lm meta acc   : (--predict-action) accuracy of the meta-action the LM itself wrote vs GT, and parse_fail rate
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import load_language_package  # noqa: E402


def rouge_l(ref: str, hyp: str) -> float:
    a, b = ref.lower().split(), hyp.lower().split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(b), lcs / len(a)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", type=int, choices=[1, 2], default=2)
    ap.add_argument("--lm", default="Qwen/Qwen1.5-1.8B-Chat")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--packets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-source", choices=["gt", "pred"], default="pred")
    ap.add_argument("--with-objects", action="store_true")
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--scene-dim", type=int, default=256)
    ap.add_argument("--traj-steps", type=int, default=6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--predict-action", action="store_true")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    lang = load_language_package()
    from importlib import import_module

    ds_mod = import_module(lang.__name__ + ".dataset")
    ds = lang.ReasonPacketDataset(
        args.labels, args.packets, meta_source=args.meta_source, with_objects=args.with_objects, lang=args.lang,
        validate=False, max_samples=args.max_samples,
    )
    print("[data]", ds.stats)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_language import build_head  # reuse the same construction

    head = build_head(args, lang, corpus_for_tiny=[r["reason"] for r in ds.rows])
    head.load(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.tiny else "cpu")
    if not (args.qlora or args.load_8bit):
        head.to(device)
    head.projector.to(device)
    head.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lang.collate_fn)
    rows, rl, cons, valid, acc12, acc_lat, lm_acc, parse_fail = [], [], [], [], [], [], [], []
    for batch in loader:
        preds = head.generate(ds_mod.move_batch(batch, device), max_new_tokens=args.max_new_tokens)
        for i, text in enumerate(preds):
            meta = int(batch["meta_action"][i])
            gt, pr = int(batch["gt_meta_action"][i]), int(batch["pred_meta_action"][i])
            lm_meta = None
            if args.predict_action:
                res = lang.parse_output(text)  # doc numbering -> back to meta_action.py numbering used here
                lm_meta = None if res.meta_idx is None else lang.to_rls_index(res.meta_idx)
                text = res.reason or ""
                parse_fail.append(float(lm_meta is None))
                lm_acc.append(float(lm_meta == gt))
                meta = gt if lm_meta is None else lm_meta
            rl.append(rouge_l(batch["reason"][i], text))
            cons.append(lang.action_consistency(text, meta))
            valid.append(float(lang.validate_reason(text)))
            acc12.append(float(gt == pr))
            acc_lat.append(float(gt // 4 == pr // 4))
            rows.append(dict(sample_token=batch["sample_token"][i], meta_used=lang.action_tag(meta), gt=lang.action_tag(gt),
                             pred=lang.action_tag(pr), label=batch["reason"][i], generated=text,
                             lm_meta=None if lm_meta is None else lang.action_tag(lm_meta)))

    def mean(x):
        return sum(x) / max(len(x), 1)

    summary = dict(n=len(rows), rougeL=mean(rl), consistency=mean(cons), valid_rate=mean(valid),
                   meta_acc_12=mean(acc12), meta_acc_lateral=mean(acc_lat))
    if args.predict_action:
        summary.update(lm_meta_acc_12=mean(lm_acc), parse_fail_rate=mean(parse_fail))
    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.out + ".summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
