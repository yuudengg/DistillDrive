"""Train the language head on cached packets + reason labels.

  stage 1 (projector only, LM frozen):
    python tools/language/train_language.py --stage 1 --lm Qwen/Qwen1.5-1.8B-Chat \
        --labels data/language/reasons_train.jsonl --packets data/language/packets_train \
        --val-labels data/language/reasons_val.jsonl --val-packets data/language/packets_val \
        --out work_dirs/language/stage1 --epochs 3 --batch-size 8 --lr 1e-3 --dtype bf16

  stage 2 (projector + LoRA):
    python tools/language/train_language.py --stage 2 --init work_dirs/language/stage1/best --load-8bit ... --lr 2e-4

  team setup = Qwen1.5-1.8B-Chat + 8bit + LoRA:  --load-8bit  (needs CUDA + bitsandbytes; pass it to eval_language.py too)
  QLoRA instead (4-bit base LM):                  --qlora
  LM also predicts the meta-action (needed by LanguageModule.infer):  add  --predict-action  (train AND eval)

  CPU smoke test (no downloads, tiny random LM):  add  --tiny  (needs packets/labels in the same format)
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import load_language_package  # noqa: E402


def build_head(args, lang, corpus_for_tiny=None):
    lora_cfg = dict(r=args.lora_r, lora_alpha=2 * args.lora_r) if args.stage == 2 else None
    common = dict(scene_dim=args.scene_dim, traj_steps=args.traj_steps, lang=args.lang, lora_cfg=lora_cfg,
                  predict_action=args.predict_action)
    if args.tiny:
        from importlib import import_module

        tiny = import_module(lang.__name__ + ".tiny")
        lm, tok = tiny.build_tiny_lm_and_tokenizer(corpus_for_tiny or ["Reason: go straight because the road is clear."])
        return lang.LanguageHead(lm, tok, projector_hidden=64, **common)
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype]
    return lang.LanguageHead.from_pretrained(
        args.lm, torch_dtype=dtype, load_in_4bit=args.qlora, load_in_8bit=args.load_8bit, **common
    )


@torch.no_grad()
def evaluate(head, loader, device, move_batch):
    head.eval()
    tot, n = 0.0, 0
    for batch in loader:
        out = head(move_batch(batch, device))
        tot += out["loss"].item() * out["num_target_tokens"].item()
        n += out["num_target_tokens"].item()
    head.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--lm", default="Qwen/Qwen1.5-1.8B-Chat")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--packets", required=True)
    ap.add_argument("--val-labels", default=None)
    ap.add_argument("--val-packets", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="checkpoint dir (projector.pt [+ lora.pt]) to start from")
    ap.add_argument("--meta-source", choices=["gt", "pred"], default="gt")
    ap.add_argument("--with-objects", action="store_true")
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--scene-dim", type=int, default=256)
    ap.add_argument("--traj-steps", type=int, default=6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--qlora", action="store_true", help="load the LM in 4-bit (stage 2)")
    ap.add_argument("--load-8bit", action="store_true", help="load the LM in 8-bit (team setup, stage 2)")
    ap.add_argument("--predict-action", action="store_true", help="LM writes '<meta-action>\nReason: ...'")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    lang = load_language_package()
    from importlib import import_module

    ds_mod = import_module(lang.__name__ + ".dataset")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.tiny else "cpu")

    train_ds = lang.ReasonPacketDataset(
        args.labels, args.packets, meta_source=args.meta_source, with_objects=args.with_objects, lang=args.lang,
        max_samples=args.max_samples,
    )
    print("[data] train", train_ds.stats)
    val_ds = None
    if args.val_labels:
        val_ds = lang.ReasonPacketDataset(
            args.val_labels, args.val_packets, meta_source=args.meta_source, with_objects=args.with_objects,
            lang=args.lang, max_samples=args.max_samples,
        )
        print("[data] val  ", val_ds.stats)

    head = build_head(args, lang, corpus_for_tiny=[r["reason"] for r in train_ds.rows] + ["Planned meta-action: Driving scene tokens:"])
    if args.init:
        head.load(args.init)
    if args.grad_ckpt:
        head.lm.gradient_checkpointing_enable()
        head.lm.enable_input_require_grads()
    if not (args.qlora or args.load_8bit):  # quantized weights are already placed on the GPU and cannot be moved
        head.to(device)
    else:
        head.projector.to(device)
    head.train()
    params = head.trainable_parameters()
    print(f"[model] trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M  device={device}")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=lang.collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=lang.collate_fn) if val_ds else None

    total_steps = args.max_steps or math.ceil(len(loader) / args.grad_accum) * args.epochs
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(s / max(total_steps, 1), 1.0)))
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    best, step, t0 = float("inf"), 0, time.time()
    run_loss, run_n = 0.0, 0
    done = False
    for epoch in range(args.epochs):
        for it, batch in enumerate(loader):
            out = head(ds_mod.move_batch(batch, device))
            (out["loss"] / args.grad_accum).backward()
            run_loss += out["loss"].item()
            run_n += 1
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    print(f"ep {epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                    run_loss, run_n = 0.0, 0
                if step >= total_steps:
                    done = True
                    break
        if val_loader is not None:
            val = evaluate(head, val_loader, device, ds_mod.move_batch)
            print(f"== epoch {epoch} val loss {val:.4f}")
            if val < best:
                best = val
                head.save(str(out_dir / "best"))
                print("   saved best")
        head.save(str(out_dir / "last"))
        if done:
            break
    print(f"[done] best val {best:.4f}  -> {out_dir}")


if __name__ == "__main__":
    main()
