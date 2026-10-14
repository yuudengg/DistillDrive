"""LanguageHead: scene packet + meta-action  ->  one-sentence reason.

    [ "Driving scene tokens:" | traj, ego, agents, map (projected) | "\\nPlanned meta-action: ...\\nReason:" ] -> " <reason><eos>"
                                                                                                                  ^ loss only here

Training stages (LLaVA style):
    stage 1: LM frozen, only SceneProjector is trained       (lora_cfg=None)
    stage 2: SceneProjector + LoRA on the LM                  (lora_cfg=dict(r=16, ...))
The driving model is never touched: the packet is detached and cached on disk.
"""
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .projector import SceneProjector
from .meta_spec import from_rls_index
from .output_parser import format_target
from .prompt import PROMPT_PREFIX, build_prompt_suffix, build_target

DEFAULT_LORA = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


def apply_lora(lm: nn.Module, lora_cfg: Optional[dict]):
    if not lora_cfg:
        return lm
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # lazily: only needed for stage 2

    if getattr(lm, "is_loaded_in_4bit", False) or getattr(lm, "is_loaded_in_8bit", False):  # QLoRA
        lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=False)

    cfg = dict(DEFAULT_LORA)
    cfg.update(lora_cfg)
    return get_peft_model(lm, LoraConfig(**cfg))


class LanguageHead(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        tokenizer,
        scene_dim: int = 256,
        traj_steps: int = 6,
        projector_hidden: int = 1024,
        lang: str = "en",
        lora_cfg: Optional[dict] = None,
        max_reason_tokens: int = 64,
        traj_scale: float = 0.1,
        predict_action: bool = False,
    ):
        super().__init__()
        for p in lm.parameters():  # freeze first, LoRA adds its own trainable weights afterwards
            p.requires_grad_(False)
        self.lm = apply_lora(lm, lora_cfg)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lang = lang
        self.max_reason_tokens = max_reason_tokens
        # True: the LM also writes the meta-action (output_parser.format_target), used by HFLanguageModule.infer
        self.predict_action = predict_action
        lm_dim = self.lm.get_input_embeddings().weight.shape[1]
        self.projector = SceneProjector(scene_dim, lm_dim, projector_hidden, traj_steps=traj_steps, traj_scale=traj_scale)
        self.config = dict(
            scene_dim=scene_dim, traj_steps=traj_steps, projector_hidden=projector_hidden, lang=lang,
            lora_cfg=lora_cfg, max_reason_tokens=max_reason_tokens, traj_scale=traj_scale,
            predict_action=predict_action,
        )

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(
        cls, lm_path: str, torch_dtype=None, device_map=None, load_in_4bit: bool = False, load_in_8bit: bool = False,
        **kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if load_in_4bit and load_in_8bit:
            raise ValueError("pick one of load_in_4bit / load_in_8bit")
        extra = {}
        if load_in_4bit or load_in_8bit:  # needs bitsandbytes + CUDA
            from transformers import BitsAndBytesConfig

            if load_in_4bit:  # QLoRA
                extra["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
                )
            else:  # 8bit + LoRA (the team's choice for Qwen1.5-1.8B-Chat)
                extra["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            device_map = device_map or {"": 0}
        try:  # transformers >= 5 prefers `dtype`, older versions only know `torch_dtype`
            lm = AutoModelForCausalLM.from_pretrained(lm_path, dtype=torch_dtype, device_map=device_map, **extra)
        except TypeError:
            lm = AutoModelForCausalLM.from_pretrained(lm_path, torch_dtype=torch_dtype, device_map=device_map, **extra)
        tok = AutoTokenizer.from_pretrained(lm_path)
        return cls(lm, tok, **kwargs)

    # ------------------------------------------------------------------ helpers
    @property
    def device(self):
        return next(self.projector.parameters()).device

    def _ids(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _embed(self, ids: List[int]) -> torch.Tensor:
        emb = self.lm.get_input_embeddings()
        return emb(torch.tensor(ids, dtype=torch.long, device=emb.weight.device))

    def _build(self, batch: Dict, with_target: bool, padding_side: str):
        """Assemble inputs_embeds / attention_mask / labels for one batch (see module docstring for the layout)."""
        dev = self.device
        lm_dtype = self.lm.get_input_embeddings().weight.dtype
        scene_emb, scene_mask = self.projector(
            batch["scene"].to(dev), batch["scene_type"].to(dev), batch["traj"].to(dev), batch["scene_mask"].to(dev)
        )
        bs = scene_emb.shape[0]
        prefix_ids = self._ids(PROMPT_PREFIX)
        prefix_emb = self._embed(prefix_ids)
        objects = batch.get("objects") or [None] * bs
        eos = self.tokenizer.eos_token_id

        seqs, masks, labels = [], [], []
        for i in range(bs):
            meta = int(batch["meta_action"][i]) if "meta_action" in batch else None
            suffix_ids = self._ids(build_prompt_suffix(None if self.predict_action else meta, objects[i], self.lang))
            parts = [prefix_emb, scene_emb[i].to(lm_dtype), self._embed(suffix_ids)]
            attn = [
                torch.ones(len(prefix_ids), dtype=torch.long, device=dev),
                scene_mask[i].long(),
                torch.ones(len(suffix_ids), dtype=torch.long, device=dev),
            ]
            lab = [torch.full((len(prefix_ids) + scene_emb.shape[1] + len(suffix_ids),), -100, dtype=torch.long, device=dev)]
            if with_target:
                if self.predict_action:  # labels / packets use meta_action.py numbering, the LM writes the doc numbering
                    text = format_target(from_rls_index(meta), batch["reason"][i])
                else:
                    text = build_target(batch["reason"][i])
                tgt = self._ids(text)[: self.max_reason_tokens] + [eos]
                parts.append(self._embed(tgt))
                attn.append(torch.ones(len(tgt), dtype=torch.long, device=dev))
                lab.append(torch.tensor(tgt, dtype=torch.long, device=dev))
            seqs.append(torch.cat(parts))
            masks.append(torch.cat(attn))
            labels.append(torch.cat(lab) if with_target else None)

        max_len = max(s.shape[0] for s in seqs)
        hid = seqs[0].shape[1]
        emb_b = seqs[0].new_zeros(bs, max_len, hid)
        mask_b = torch.zeros(bs, max_len, dtype=torch.long, device=dev)
        lab_b = torch.full((bs, max_len), -100, dtype=torch.long, device=dev) if with_target else None
        for i, (s, m) in enumerate(zip(seqs, masks)):
            n = s.shape[0]
            sl = slice(0, n) if padding_side == "right" else slice(max_len - n, max_len)
            emb_b[i, sl], mask_b[i, sl] = s, m
            if with_target:
                lab_b[i, sl] = labels[i]
        return emb_b, mask_b, lab_b

    # ------------------------------------------------------------------ train / eval loss
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        emb, mask, labels = self._build(batch, with_target=True, padding_side="right")
        out = self.lm(inputs_embeds=emb, attention_mask=mask, labels=labels)
        return {"loss": out.loss, "num_target_tokens": (labels != -100).sum()}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, batch: Dict, max_new_tokens: int = 48, **gen_kwargs) -> List[str]:
        emb, mask, _ = self._build(batch, with_target=False, padding_side="left")
        gen_kwargs.setdefault("do_sample", False)
        out = self.lm.generate(
            inputs_embeds=emb,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        # with inputs_embeds only, generate() returns just the new tokens
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]

    # ------------------------------------------------------------------ params / checkpoints
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _lora_state(self):
        return {k: v.detach().cpu() for k, v in self.lm.state_dict().items() if "lora_" in k}

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
        lora = self._lora_state()
        if lora:
            torch.save(lora, os.path.join(out_dir, "lora.pt"))
        with open(os.path.join(out_dir, "language_head.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    def load(self, ckpt_dir: str, strict: bool = True):
        self.projector.load_state_dict(torch.load(os.path.join(ckpt_dir, "projector.pt"), map_location="cpu"), strict=strict)
        lora_path = os.path.join(ckpt_dir, "lora.pt")
        if os.path.exists(lora_path):
            missing = self.lm.load_state_dict(torch.load(lora_path, map_location="cpu"), strict=False)
            unexpected = list(missing.unexpected_keys)
            if unexpected:
                raise RuntimeError(f"LoRA keys do not match this model (lora_cfg differs?): {unexpected[:3]}")
        return self
