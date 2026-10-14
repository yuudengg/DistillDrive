"""Prompt / label-text utilities for the language branch (pure python, no torch needed).

The 12 meta-actions follow models/motion/meta_action.py:
    meta_action = lateral * 4 + longitudinal
    lateral      : 0 right, 1 left, 2 straight
    longitudinal : 0 accelerate, 1 keep, 2 decelerate, 3 stop
(tests/language/test_language_pipeline.py checks that these tables stay in sync.)
"""
import re
from typing import Optional, Sequence, Tuple

LATERAL_NAMES = ("right", "left", "straight")
LONGITUDINAL_NAMES = ("accelerate", "keep", "decelerate", "stop")
NUM_META_ACTIONS = len(LATERAL_NAMES) * len(LONGITUDINAL_NAMES)  # 12

# nuScenes detection classes in SparseDrive order
CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)

_LAT_PHRASE = {
    "en": {"right": "turn right", "left": "turn left", "straight": "go straight"},
    "ko": {"right": "우회전", "left": "좌회전", "straight": "직진"},
}
_LON_PHRASE = {
    "en": {
        "accelerate": "while speeding up",
        "keep": "at a steady speed",
        "decelerate": "while slowing down",
        "stop": "and come to a stop",
    },
    "ko": {"accelerate": "가속", "keep": "속도 유지", "decelerate": "감속", "stop": "정지"},
}
_HEADERS = {
    "en": dict(prefix="Driving scene tokens:", objects="Nearby objects", action="Planned meta-action", reason="Reason:"),
    "ko": dict(prefix="Driving scene tokens:", objects="주변 객체", action="계획된 행동", reason="근거:"),
}

PROMPT_PREFIX = _HEADERS["en"]["prefix"]  # same in every language (scene tokens follow it)


# ----------------------------------------------------------------------------------
# meta-action <-> text
# ----------------------------------------------------------------------------------
def split_meta_action(meta_action: int) -> Tuple[int, int]:
    meta_action = int(meta_action)
    if not 0 <= meta_action < NUM_META_ACTIONS:
        raise ValueError(f"meta_action must be in [0, {NUM_META_ACTIONS}), got {meta_action}")
    return meta_action // len(LONGITUDINAL_NAMES), meta_action % len(LONGITUDINAL_NAMES)


def action_tag(meta_action: int) -> str:
    """6 -> 'decelerate+left' (same string as MetaActionHead.to_text)."""
    lat, lon = split_meta_action(meta_action)
    return f"{LONGITUDINAL_NAMES[lon]}+{LATERAL_NAMES[lat]}"


def action_phrase(meta_action: int, lang: str = "en") -> str:
    """6 -> 'turn left while slowing down' / '좌회전 + 감속'."""
    lat, lon = split_meta_action(meta_action)
    lat_name, lon_name = LATERAL_NAMES[lat], LONGITUDINAL_NAMES[lon]
    if lang == "ko":
        return f"{_LAT_PHRASE['ko'][lat_name]} + {_LON_PHRASE['ko'][lon_name]}"
    return f"{_LAT_PHRASE['en'][lat_name]} {_LON_PHRASE['en'][lon_name]}"


# ----------------------------------------------------------------------------------
# prompt for the language model (text around the scene tokens)
# ----------------------------------------------------------------------------------
def build_prompt_suffix(meta_action: Optional[int], objects: Optional[str] = None, lang: str = "en") -> str:
    """Text placed AFTER the scene tokens.

    meta_action given -> ends with the reason header (the model writes only the reason).
    meta_action None  -> ends with a newline; the model writes output_parser.format_target(...) itself.
    """
    h = _HEADERS[lang]
    text = ""
    if objects:
        text += f"\n{h['objects']}: {objects}"
    if meta_action is None:
        return text + "\n"
    text += f"\n{h['action']}: {action_tag(meta_action)} ({action_phrase(meta_action, lang)})"
    text += f"\n{h['reason']}"
    return text


def build_target(reason: str) -> str:
    return " " + reason.strip()


# ----------------------------------------------------------------------------------
# optional: textual object list from detected agents (ablation lever, off by default)
# ----------------------------------------------------------------------------------
def _to_list(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def describe_objects(
    xy: Sequence,
    labels: Sequence,
    conf: Sequence,
    topn: int = 5,
    max_dist: float = 40.0,
    min_conf: float = 0.3,
) -> Optional[str]:
    """'car 12m ahead, 3m left; pedestrian 5m ahead, 2m right'.

    ASSUMPTION: xy is in the nuScenes LiDAR ego frame (x = right, y = forward), the same frame in which
    MetaActionHead treats x >= 2m as a right turn. Check once on a real sample.
    """
    xy, labels, conf = _to_list(xy), _to_list(labels), _to_list(conf)
    items = []
    for (x, y), lab, c in zip(xy, labels, conf):
        if c < min_conf:
            continue
        dist = (x * x + y * y) ** 0.5
        if dist > max_dist:
            continue
        name = CLASS_NAMES[int(lab)].replace("_", " ")
        fwd = "ahead" if y >= 0 else "behind"
        side = "right" if x >= 0 else "left"
        items.append((dist, f"{name} {abs(round(y))}m {fwd}, {abs(round(x))}m {side}"))
    items.sort(key=lambda t: t[0])
    if not items:
        return None
    return "; ".join(text for _, text in items[:topn])


# ----------------------------------------------------------------------------------
# label generation prompt (for the offline vision-language model, e.g. Qwen2.5-VL)
# ----------------------------------------------------------------------------------
def build_label_prompt(meta_action: int, lang: str = "en", max_words: int = 30) -> str:
    phrase = action_phrase(meta_action, lang)
    if lang == "ko":
        return (
            "당신은 숙련된 운전 교관입니다. 차량 전방 카메라 영상을 보고 있습니다.\n"
            f"운전자는 곧 다음 행동을 합니다: {phrase}.\n"
            f"이 행동이 왜 적절한지 {max_words}단어 이내의 한 문장으로 설명하세요. "
            "영상에서 분명히 보이는 것(신호등, 다른 차량, 보행자, 차선, 도로 구조)만 언급하고, "
            "카메라나 영상이라는 말은 쓰지 말며, 보이지 않는 객체를 지어내지 마세요. "
            "바로 이유부터 시작하세요."
        )
    return (
        "You are an experienced driving instructor looking at the front camera image of a car.\n"
        f"The driver is about to: {phrase}.\n"
        f"In ONE sentence of at most {max_words} words, explain why this action is appropriate. "
        "Mention only things that are clearly visible (traffic lights, other vehicles, pedestrians, "
        "lane markings, road layout). Do not mention the camera or the image, and do not invent objects. "
        "Start directly with the reason."
    )


# ----------------------------------------------------------------------------------
# label cleaning / validation / consistency
# ----------------------------------------------------------------------------------
_BANNED = ("the image", "this image", "the camera", "the picture", "the photo", "i cannot", "i can't", "sorry", "as an ai")


def clean_reason(text: str) -> str:
    text = re.sub(r"^\s*(reason|근거)\s*[:：]\s*", "", text.strip(), flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_reason(reason: str, min_words: int = 4, max_words: int = 60) -> bool:
    reason = reason.strip()
    n = len(reason.split())
    if not (min_words <= n <= max_words):
        return False
    low = reason.lower()
    return not any(b in low for b in _BANNED)


_LAT_KEYS = {
    "right": ("right",),
    "left": ("left",),
    "straight": ("straight", "ahead", "forward", "continue", "proceed", "직진", "앞"),
}
_LON_KEYS = {
    "accelerate": ("accelerat", "speed up", "speeding up", "faster", "가속"),
    "keep": ("steady", "constant", "maintain", "keep", "same speed", "유지"),
    "decelerate": ("slow", "decelerat", "brak", "reduc", "감속"),
    "stop": ("stop", "halt", "wait", "yield", "정지", "멈"),
}


def action_consistency(reason: str, meta_action: int) -> float:
    """Soft metric in {0, 0.5, 1}: does the reason mention the lateral / longitudinal part of the action?"""
    lat, lon = split_meta_action(meta_action)
    low = reason.lower()
    lat_ok = any(k in low for k in _LAT_KEYS[LATERAL_NAMES[lat]])
    lon_ok = any(k in low for k in _LON_KEYS[LONGITUDINAL_NAMES[lon]])
    return (float(lat_ok) + float(lon_ok)) / 2.0
