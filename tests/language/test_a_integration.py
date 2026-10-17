"""CPU tests for what motion_planning_head (A/B feedback, 877c1f6 + tools/language/patches) uses from C (no mmcv needed):
  * LangPacketBuilder is registered in PLUGIN_LAYERS (the head builds it with build(lang_packet_cfg, PLUGIN_LAYERS))
  * build_language_module with the config types B's code passes (mock / LanguageModule)
  * packet -> _scene_input (split_packet, sample b) -> infer_batch, with failures returned as None
  * a module loaded from config goes to the GPU without device=...

    pytest tests/language -q
"""
import subprocess
import sys
import textwrap

import torch

from test_language_pipeline import ROOT, lang, make_builder, make_inputs


def test_lang_packet_builder_registered_in_plugin_layers():
    code = textwrap.dedent(f"""
        import sys, types
        import torch.nn as nn
        seen = {{}}
        class Registry:
            def register_module(self, name=None, force=False, module=None):
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
        assert seen["LangPacketBuilder"] is lang.LangPacketBuilder, sorted(seen)
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_build_language_module_from_feedback_config():
    assert lang.build_language_module(None) is None  # enabled=False path never builds it, None also means off
    m = lang.build_language_module(dict(type="mock", mode="cycle", fail_every=4))
    assert isinstance(m, lang.MockLanguageModule)
    assert lang.language_module._REGISTRY["LanguageModule"] is lang.HFLanguageModule


def test_packet_to_query_language_with_failures():
    """What _query_language does after the patch: one packet, then sample b of it for every triggered sample."""
    pkt = make_builder()(**make_inputs(seed=3))
    triggered = [0, 2]
    lm = lang.MockLanguageModule(mode="cycle", fail_every=2)
    inputs = [lang.split_packet(pkt, b) for b in triggered]
    assert all(x["traj"].dim() == 2 and "agent_xy" in x for x in inputs)  # no batch dim, packet keys
    results = lm.infer_batch(inputs)
    assert results[0][0] == 0 and results[1] == (None, None)  # the 2nd call fails -> B keeps that sample's 1st pass


def test_hf_module_loaded_from_config_goes_to_gpu(monkeypatch):
    """build_language_module(dict(type="LanguageModule", ...)) without device=... must not silently stay on the CPU."""
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
    lang.build_language_module(dict(type="LanguageModule", ckpt_dir="x", lm_path="y"))
    assert moved == ["cuda"]
    head2, moved2 = tiny_head()
    lang.LanguageModule(ckpt_dir="x", lm_path="y", load_in_8bit=True)  # quantized: already placed, never moved
    lang.LanguageModule(head=head2)  # injected head (tests / notebooks): left where it is
    assert moved == ["cuda"] and moved2 == []
