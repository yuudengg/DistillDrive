"""Language module interface (owned by C).

    module = build_language_module(cfg)
    meta_idx, reason = module.infer(scene_input)          # one sample
    results = module.infer_batch([scene_input, ...])      # list of (meta_idx, reason)

Contract for callers (A's trigger code):
  * meta_idx is an int in [0, 12) following meta_spec.py. HF module: if the LM output cannot be parsed it falls back to
    the rule meta-action in scene_input ('meta_action' = meta_action.py numbering, or 'meta_idx'); None only if neither exists
  * reason is a str, or None on failure
  * infer() never raises on bad LM output -> the caller decides the fallback (e.g. keep the 1st-pass trajectory)
  * no gradient flows out of this module

scene_input (NOT FINAL - the team still decides 'text summary vs query embedding'):
  a dict; keys a module may read:
    "scene"      : [L, D] float   scene tokens (ego / agents / map) from the dump
    "scene_mask" : [L] bool
    "scene_type" : [L] long       0 ego, 1 agent, 2 map
    "traj"       : [T, 2] float   1st-pass trajectory (positions)
    "text"       : str            text summary of the scene (if the text route is chosen)
    "meta_idx"   : int            rule-based meta-action from B (optional, used by MockLanguageModule 'echo')
  Unknown keys are ignored, so A and B can already pass whatever they have.
"""
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .meta_spec import NUM_META, from_rls_index, meta_index, meta_name
from .output_parser import ParseResult, parse_output

InferResult = Tuple[Optional[int], Optional[str]]


class LanguageModuleBase:
    """Every language module (mock or real) implements infer(). infer_batch() loops by default."""

    def infer(self, scene_input: Dict) -> InferResult:
        raise NotImplementedError

    def infer_batch(self, scene_inputs: Sequence[Dict]) -> List[InferResult]:
        return [self.infer(x) for x in scene_inputs]


# ----------------------------------------------------------------------------------
# Mock: for A (trigger / feedback wiring) and B (adaLN, rollback) - no torch, no LM
# ----------------------------------------------------------------------------------
class MockLanguageModule(LanguageModuleBase):
    """Returns fixed / predictable answers so the integration can be tested on CPU.

    mode:
      "fixed" : always (meta_idx, reason)
      "cycle" : 0, 1, ..., 11, 0, ... -> exercises every adaLN embedding row
      "echo"  : returns scene_input["meta_idx"] if present (e.g. B's rule output), else the fixed meta_idx
    fail_every: every N-th call returns (None, None) like an unparsable LM output -> tests A's fallback path
    delay_s   : sleep per call -> lets A see how a slow LM affects the trigger loop
    raw_output: if given, run this exact text through the real parser instead (tests the parse path end to end)
    """

    def __init__(
        self,
        mode: str = "fixed",
        meta_idx: int = meta_index("straight", "keep"),
        reason: Optional[str] = None,
        fail_every: int = 0,
        delay_s: float = 0.0,
        raw_output: Optional[str] = None,
    ):
        assert mode in ("fixed", "cycle", "echo"), mode
        if not 0 <= int(meta_idx) < NUM_META:
            raise ValueError(f"meta_idx must be in [0, {NUM_META})")
        self.mode = mode
        self.meta_idx = int(meta_idx)
        self.reason = reason
        self.fail_every = fail_every
        self.delay_s = delay_s
        self.raw_output = raw_output
        self.num_calls = 0
        self.calls: List[Dict] = []  # scene_input keys seen per call, for assertions in tests

    def _reason_for(self, meta_idx: int) -> str:
        return self.reason or f"[mock] chose {meta_name(meta_idx)} for testing."

    def infer(self, scene_input: Dict) -> InferResult:
        self.num_calls += 1
        self.calls.append({"keys": sorted(scene_input.keys()) if isinstance(scene_input, dict) else None})
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_every and self.num_calls % self.fail_every == 0:
            return None, None
        if self.raw_output is not None:
            res = parse_output(self.raw_output)
            return res.meta_idx, res.reason

        if self.mode == "cycle":
            meta = (self.num_calls - 1) % NUM_META
        elif self.mode == "echo" and isinstance(scene_input, dict) and scene_input.get("meta_idx") is not None:
            meta = int(scene_input["meta_idx"])
        else:
            meta = self.meta_idx
        return meta, self._reason_for(meta)


# ----------------------------------------------------------------------------------
# Real module (skeleton): LM predicts "Meta-action: ...\nReason: ..." and the parser reads it back
# ----------------------------------------------------------------------------------
class HFLanguageModule(LanguageModuleBase):
    """Qwen (+ LoRA adapter; team default Qwen1.5-1.8B-Chat loaded in 8bit) behind the same interface.

    Flow:  scene_input -> LanguageHead (projector + LM, prompt does NOT reveal the meta-action) -> generate -> parse_output
    Train the head with  train_language.py --predict-action  (target = output_parser.format_target(meta, reason)).

    Only the embedding route (scene / scene_mask / scene_type / traj) is implemented.
    TODO if the team picks the text route: build the prompt from scene_input["text"] instead.
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        lm_path: Optional[str] = None,
        head=None,  # an already built LanguageHead (tests / notebooks); otherwise loaded from ckpt_dir + lm_path
        lenient: bool = False,
        max_new_tokens: int = 64,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: Optional[str] = None,
    ):
        if head is None:
            head = self._load_head(ckpt_dir, lm_path, load_in_4bit, load_in_8bit)
        assert head.predict_action, "this checkpoint was trained without --predict-action: it cannot output a meta-action"
        if device is not None and not (load_in_4bit or load_in_8bit):  # quantized weights are already on the GPU
            head.to(device)
        self.head = head.eval()
        self.lenient, self.max_new_tokens = lenient, max_new_tokens
        self.last_parse: List[ParseResult] = []
        self.num_fallback = 0

    @staticmethod
    def _load_head(ckpt_dir: str, lm_path: str, load_in_4bit: bool, load_in_8bit: bool = False):
        import json
        import os

        import torch

        from .language_head import LanguageHead

        with open(os.path.join(ckpt_dir, "language_head.json")) as f:
            cfg = json.load(f)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        head = LanguageHead.from_pretrained(
            lm_path, torch_dtype=dtype, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, **cfg
        )
        return head.load(ckpt_dir)

    def infer(self, scene_input: Dict) -> InferResult:
        return self.infer_batch([scene_input])[0]

    def infer_batch(self, scene_inputs: Sequence[Dict]) -> List[InferResult]:
        import torch

        batch = {k: torch.stack([torch.as_tensor(x[k]) for x in scene_inputs]) for k in ("scene", "scene_mask", "scene_type", "traj")}
        batch["scene"], batch["traj"] = batch["scene"].float(), batch["traj"].float()
        texts = self.head.generate(batch, max_new_tokens=self.max_new_tokens)
        self.last_parse = [parse_output(t, lenient=self.lenient) for t in texts]
        out = []
        for x, p in zip(scene_inputs, self.last_parse):
            meta = p.meta_idx
            if meta is None:  # "tagged freedom": the tag is mandatory, so a missing tag -> rule meta-action
                self.num_fallback += 1
                meta = _rule_meta(x)
            out.append((meta, p.reason))
        return out


def _rule_meta(scene_input: Dict) -> Optional[int]:
    """Rule meta-action carried by the input: packet 'meta_action' (meta_action.py numbering) or 'meta_idx' (meta_spec)."""
    if scene_input.get("meta_idx") is not None:
        return int(scene_input["meta_idx"])
    if scene_input.get("meta_action") is not None:
        return from_rls_index(int(scene_input["meta_action"]))
    return None


LanguageModule = HFLanguageModule  # name used in the role documents


# ----------------------------------------------------------------------------------
# factory (config-driven, so A only writes a dict in the config)
# ----------------------------------------------------------------------------------
_REGISTRY = {"mock": MockLanguageModule, "hf": HFLanguageModule}


def build_language_module(cfg: Optional[Dict]) -> Optional[LanguageModuleBase]:
    """cfg=None -> None (language module off, the default).
    cfg=dict(type="mock", mode="cycle") / dict(type="hf", ckpt_dir=..., lm_path=...)
    """
    if cfg is None:
        return None
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in _REGISTRY:
        raise KeyError(f"unknown language module type '{kind}', choose from {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**cfg)
