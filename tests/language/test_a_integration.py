"""CPU tests for what A's motion_planning_head uses from C (no mmcv needed):
  * LanguageModule / MockLanguageModule are registered in PLUGIN_LAYERS (A builds them with build_from_cfg)
  * feedback_meta / select_per_sample: per-sample fallback to the 1st pass when infer_batch returns None
  * packet -> split_packet -> infer_batch works with A's per-sample call

    pytest tests/language -q
"""
import subprocess
import sys
import textwrap

import torch

from test_language_pipeline import ROOT, lang, make_builder, make_inputs


def test_language_modules_registered_in_plugin_layers():
    code = textwrap.dedent(f"""
        import sys, types
        import torch.nn as nn
        seen = {{}}
        class Registry:
            def register_module(self, name=None, force=False, module=None):
                if module is not None:
                    seen[name or module.__name__] = module
                    return module
                def deco(cls):
                    seen[name or cls.__name__] = cls
                    return cls
                return deco
        for n in ("mmcv", "mmcv.runner", "mmcv.runner.base_module", "mmcv.cnn", "mmcv.cnn.bricks", "mmcv.cnn.bricks.registry"):
            sys.modules[n] = types.ModuleType(n)
        sys.modules["mmcv.runner.base_module"].BaseModule = nn.Module
        sys.modules["mmcv.cnn.bricks.registry"].PLUGIN_LAYERS = Registry()
        sys.path.insert(0, r"{ROOT / 'tools' / 'language'}")
        from _compat import load_language_package
        lang = load_language_package(r"{ROOT}")
        assert seen["LanguageModule"] is lang.HFLanguageModule, sorted(seen)
        assert seen["MockLanguageModule"] is lang.MockLanguageModule, sorted(seen)
        assert seen["LangPacketBuilder"] is lang.LangPacketBuilder, sorted(seen)
        m = seen["MockLanguageModule"](mode="cycle", fail_every=4)  # what build_from_cfg does with A's config
        assert m.infer({{}})[0] == 0
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_feedback_meta_and_select_per_sample():
    ok, meta = lang.feedback_meta([(5, "a"), (None, None), (2, None)])
    assert ok.tolist() == [True, False, True] and meta.tolist() == [5, 0, 2]

    first = (torch.zeros(3, 1, 18), torch.zeros(3, 1, 18, 6, 2))  # e.g. plan_cls, plan_reg
    second = (torch.ones(3, 1, 18), torch.ones(3, 1, 18, 6, 2))
    out = lang.select_per_sample(ok, first, second)
    assert isinstance(out, tuple)
    assert out[0][:, 0, 0].tolist() == [1, 0, 1] and out[1][:, 0, 0, 0, 0].tolist() == [1, 0, 1]


def test_packet_to_infer_batch_with_failures():
    pkt = make_builder()(**make_inputs(seed=3))
    bs = pkt["traj"].shape[0]
    lm = lang.MockLanguageModule(mode="cycle", fail_every=2)
    results = lm.infer_batch([lang.split_packet(pkt, b) for b in range(bs)])
    ok, meta = lang.feedback_meta(results)
    assert ok.tolist() == [True, False, True][:bs] and meta.shape == (bs,)


def test_hf_module_loaded_from_config_goes_to_gpu(monkeypatch):
    """A builds the module from config without device=... -> it must not silently stay on the CPU."""
    from test_language_pipeline import REASONS, tiny

    lm_mod = sys.modules[lang.__name__ + ".language_module"]

    def tiny_head():
        lm, tok = tiny.build_tiny_lm_and_tokenizer(REASONS)
        head = lang.LanguageHead(lm, tok, predict_action=True, input_mode="text")
        moved = []
        head.to = lambda dev: moved.append(dev) or head
        return head, moved

    head, moved = tiny_head()
    monkeypatch.setattr(lm_mod.HFLanguageModule, "_load_head", staticmethod(lambda *a, **k: head))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    lang.LanguageModule(ckpt_dir="x", lm_path="y")  # what build_from_cfg does with A's config
    assert moved == ["cuda"]
    head2, moved2 = tiny_head()
    lang.LanguageModule(ckpt_dir="x", lm_path="y", load_in_8bit=True)  # quantized: already placed, never moved
    lang.LanguageModule(head=head2)  # injected head (tests / notebooks): left where it is
    assert moved == ["cuda"] and moved2 == []
