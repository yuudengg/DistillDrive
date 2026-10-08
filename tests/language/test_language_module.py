"""Tests for C's first-week deliverables: meta_spec, output_parser, language_module (mock + factory).

    pytest tests/language/test_language_module.py -q

No torch model, no transformers download, no GPU.
"""
import sys
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "language"))
from _compat import load_language_package  # noqa: E402

lang = load_language_package(ROOT)
spec = import_module(lang.__name__ + ".meta_spec")
parser = import_module(lang.__name__ + ".output_parser")
lm_mod = import_module(lang.__name__ + ".language_module")


# ---------------------------------------------------------------------------- meta_spec
def test_numbering_follows_the_doc():
    # doc: lateral 0 left / 1 straight / 2 right, longitudinal 0 acc / 1 keep / 2 dec / 3 stop, idx = lat*4 + lon
    assert spec.LATERAL == ("left", "straight", "right")
    assert spec.LONGITUDINAL == ("accelerate", "keep", "decelerate", "stop")
    assert spec.meta_index("left", "accelerate") == 0
    assert spec.meta_index("straight", "keep") == 5
    assert spec.meta_index("right", "stop") == 11
    names = [spec.meta_name(i) for i in range(12)]
    assert len(set(names)) == 12
    for i in range(12):
        lat, lon = spec.split_meta(i)
        assert spec.meta_index(spec.LATERAL[lat], spec.LONGITUDINAL[lon]) == i
    assert spec.meta_name_ko(6) == "직진+감속"
    with pytest.raises(ValueError):
        spec.split_meta(12)


def test_conversion_from_teammate_numbering():
    # meta_action.py: lateral 0 right / 1 left / 2 straight
    assert spec.from_rls_index(6) == spec.meta_index("left", "decelerate")  # rls 6 = left + decelerate
    assert spec.from_rls_index(0) == spec.meta_index("right", "accelerate")
    assert spec.from_rls_index(9) == spec.meta_index("straight", "keep")
    assert sorted(spec.from_rls_index(i) for i in range(12)) == list(range(12))
    for i in range(12):
        assert spec.to_rls_index(spec.from_rls_index(i)) == i


# ---------------------------------------------------------------------------- output parser
REASON = "A pedestrian is crossing ahead, so the car slows down."


@pytest.mark.parametrize("idx", range(12))
def test_format_then_parse_roundtrip(idx):
    text = parser.format_target(idx, REASON)
    res = parser.parse_output(text)
    assert res.ok and res.meta_idx == idx and res.reason == REASON and not res.partial


@pytest.mark.parametrize(
    "text, expected",
    [
        ("meta-action: STRAIGHT+DECELERATE\nreason: ped ahead.", 6),
        ("Meta-action: decelerate+straight\nReason: ped ahead.", 6),  # reversed order
        ("Meta-action: turn left while slowing down\nReason: junction.", 2),  # phrase
        ("Meta action: go straight and speed up\nReason: road clear.", 4),
        ("메타행동: 좌회전+감속\n근거: 교차로에서 보행자가 있다.", 2),  # Korean
        ("행동: 우+정지\n이유: 신호가 빨간불이다.", 11),  # Korean short form
        ("```\nMeta-action: right+keep\nReason: lane change.\n```", 9),  # code fence
        ('"Meta-action: left+stop\nReason: red light."', 3),  # quotes
        ("Sure! Meta-action: straight+keep\nReason: clear road. Hope this helps.", 5),  # chatter before
    ],
)
def test_parser_tolerates_real_world_outputs(text, expected):
    res = parser.parse_output(text)
    assert res.ok, res.error
    assert res.meta_idx == expected


def test_parser_reports_failures_without_raising():
    for bad in ["", "   ", None, "blah blah", "Meta-action: banana\nReason: x"]:
        res = parser.parse_output(bad)
        assert not res.ok and res.meta_idx is None and res.error
    amb = parser.parse_output("Meta-action: left or right, slow down\nReason: unsure.")
    assert not amb.ok and "ambiguous" in amb.error
    no_reason = parser.parse_output("Meta-action: straight+keep")
    assert no_reason.meta_idx == 5 and no_reason.reason is None and not no_reason.ok
    assert "reason" in no_reason.error


def test_reason_without_header_and_multiline_reason():
    res = parser.parse_output("Meta-action: right+decelerate\nThe car ahead is braking.")
    assert res.ok and res.meta_idx == 10 and res.reason == "The car ahead is braking."
    res = parser.parse_output("Meta-action: left+keep\nReason: The road curves\n   to the left.")
    assert res.reason == "The road curves to the left."


def test_lenient_mode_fills_one_missing_axis():
    strict = parser.parse_output("Meta-action: stop\nReason: red light.")
    assert not strict.ok
    lenient = parser.parse_output("Meta-action: stop\nReason: red light.", lenient=True)
    assert lenient.ok and lenient.partial and lenient.meta_idx == spec.meta_index("straight", "stop")
    lenient2 = parser.parse_output("Meta-action: turn left\nReason: junction.", lenient=True)
    assert lenient2.meta_idx == spec.meta_index("left", "keep") and lenient2.partial
    # ambiguity is never "fixed" by lenient mode
    assert not parser.parse_output("Meta-action: left right\nReason: x.", lenient=True).ok


def test_korean_single_syllables_need_to_stand_alone():
    # '우' inside '우산' (umbrella) must not count as 'right'
    res = parser.parse_output("행동: 직진+유지\n근거: 우산을 쓴 보행자가 인도에 있다.")
    assert res.ok and res.meta_idx == 5


# ---------------------------------------------------------------------------- mock module + factory
def test_mock_fixed_and_contract():
    m = lm_mod.MockLanguageModule()
    meta, reason = m.infer({"scene": None, "traj": None})
    assert meta == spec.meta_index("straight", "keep") and isinstance(reason, str)
    assert m.num_calls == 1 and m.calls[0]["keys"] == ["scene", "traj"]


def test_mock_cycle_covers_all_classes():
    m = lm_mod.MockLanguageModule(mode="cycle")
    metas = [m.infer({})[0] for _ in range(24)]
    assert metas[:12] == list(range(12)) and metas[12:] == list(range(12))


def test_mock_echo_uses_rule_meta_when_given():
    m = lm_mod.MockLanguageModule(mode="echo", meta_idx=0)
    assert m.infer({"meta_idx": 7})[0] == 7
    assert m.infer({})[0] == 0


def test_mock_failures_and_raw_output():
    m = lm_mod.MockLanguageModule(fail_every=3)
    out = [m.infer({}) for _ in range(6)]
    assert out[2] == (None, None) and out[5] == (None, None) and out[0][0] is not None
    raw = lm_mod.MockLanguageModule(raw_output="Meta-action: right+stop\nReason: red light ahead.")
    assert raw.infer({}) == (11, "red light ahead.")
    broken = lm_mod.MockLanguageModule(raw_output="I am not sure.")
    assert broken.infer({}) == (None, None)


def test_infer_batch_and_factory():
    assert lm_mod.build_language_module(None) is None  # default off
    m = lm_mod.build_language_module(dict(type="mock", mode="cycle"))
    assert isinstance(m, lm_mod.MockLanguageModule)
    res = m.infer_batch([{}, {}, {}])
    assert [r[0] for r in res] == [0, 1, 2]
    with pytest.raises(KeyError):
        lm_mod.build_language_module(dict(type="nope"))
    with pytest.raises(FileNotFoundError):  # hf needs a real checkpoint (see test_hf_language_module.py)
        lm_mod.build_language_module(dict(type="hf", ckpt_dir="x", lm_path="y"))
    with pytest.raises(ValueError):
        lm_mod.MockLanguageModule(meta_idx=12)
