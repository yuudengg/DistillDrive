"""LM output format and parser.

Target text the LM is trained to produce (and that label files are turned into):

    Meta-action: straight+decelerate
    Reason: A pedestrian is crossing ahead, so the car slows down.

parse_output() reads that back into (meta_idx, reason). It is tolerant of what small LMs actually emit:
upper/lower case, the two axes in either order ("decelerate+straight"), phrases ("turn left while slowing
down"), Korean ("좌회전+감속", "근거: ..."), code fences, quotes and trailing chatter.
It never raises: a failure is reported in ParseResult.ok / .error so the caller can fall back.
"""
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .meta_spec import LATERAL, LONGITUDINAL, NUM_LONGITUDINAL, meta_name

META_HEADER = "Meta-action:"
REASON_HEADER = "Reason:"

# keyword -> canonical name. Longer phrases are matched first, matching is case-insensitive.
_LATERAL_KEYS = {
    "left": ("turn left", "left turn", "turning left", "left", "좌회전", "좌"),
    "straight": ("go straight", "going straight", "straight", "직진", "직"),
    "right": ("turn right", "right turn", "turning right", "right", "우회전", "우"),
}
_LONGITUDINAL_KEYS = {
    "accelerate": ("speed up", "speeding up", "accelerating", "accelerate", "가속"),
    "keep": ("keep speed", "steady speed", "constant speed", "maintain speed", "steady", "keep", "maintain", "속도 유지", "유지"),
    "decelerate": ("slow down", "slowing down", "decelerating", "decelerate", "brake", "braking", "감속"),
    "stop": ("come to a stop", "stopping", "stop", "halt", "정지", "멈춤"),
}

_META_LINE = re.compile(r"(?:meta[\s_-]*action|메타\s*행동|행동)\s*[:：]\s*(?P<body>[^\n]*)", re.IGNORECASE)
_REASON_LINE = re.compile(r"(?:reason|근거|이유)\s*[:：]\s*(?P<body>.*)", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"\b([a-z]+)\s*\+\s*([a-z]+)\b", re.IGNORECASE)


@dataclass
class ParseResult:
    meta_idx: Optional[int]
    reason: Optional[str]
    ok: bool
    error: Optional[str] = None
    partial: bool = False  # True if one axis was filled with a default (lenient mode)


def format_target(meta_idx: int, reason: str) -> str:
    """Training target / label text. Inverse of parse_output."""
    return f"{META_HEADER} {meta_name(meta_idx)}\n{REASON_HEADER} {reason.strip()}"


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)  # code fences
    return text.strip().strip('"').strip("'").strip()


def _find_axis(text: str, table) -> Tuple[Optional[str], Optional[str]]:
    """Return (name, error). Korean single-syllable keys only count as whole tokens."""
    low = text.lower()
    found = set()
    for name, keys in table.items():
        for k in sorted(keys, key=len, reverse=True):
            if re.fullmatch(r"[가-힣]", k):  # '좌', '직', '우' -> must stand alone (e.g. '좌+감속')
                pattern = rf"(?<![가-힣]){k}(?![가-힣])"
            elif re.search(r"[가-힣]", k):
                pattern = re.escape(k)
            else:
                pattern = rf"\b{re.escape(k)}\b"
            if re.search(pattern, low):
                found.add(name)
                break
    if not found:
        return None, "not found"
    if len(found) > 1:
        return None, f"ambiguous ({', '.join(sorted(found))})"
    return found.pop(), None


def _parse_meta(body: str):
    # 1) canonical tag in either order: "left+decelerate" / "decelerate+left"
    for a, b in _TAG.findall(body):
        a, b = a.lower(), b.lower()
        if a in LATERAL and b in LONGITUDINAL:
            return a, b, None
        if b in LATERAL and a in LONGITUDINAL:
            return b, a, None
    # 2) keywords / phrases / Korean
    lat, lat_err = _find_axis(body, _LATERAL_KEYS)
    lon, lon_err = _find_axis(body, _LONGITUDINAL_KEYS)
    err = None
    if lat_err or lon_err:
        err = "; ".join(e for e in (f"lateral {lat_err}" if lat_err else None, f"longitudinal {lon_err}" if lon_err else None) if e)
    return lat, lon, err


def parse_output(text: str, lenient: bool = False) -> ParseResult:
    """LM output text -> ParseResult. Never raises.

    lenient=True: if exactly one axis is missing (not ambiguous), fill it with 'straight' / 'keep'
                  and mark partial=True. Default is strict.
    """
    if not isinstance(text, str) or not text.strip():
        return ParseResult(None, None, False, "empty output")
    text = _clean(text)

    m = _META_LINE.search(text)
    meta_body = m.group("body") if m else text.splitlines()[0]
    lat, lon, err = _parse_meta(meta_body)

    partial = False
    if lenient and err:
        missing_lat = lat is None and "lateral not found" in err
        missing_lon = lon is None and "longitudinal not found" in err
        if missing_lat and lon is not None:
            lat, err, partial = "straight", None, True
        elif missing_lon and lat is not None:
            lon, err, partial = "keep", None, True

    meta_idx = None
    if lat is not None and lon is not None and not err:
        meta_idx = LATERAL.index(lat) * NUM_LONGITUDINAL + LONGITUDINAL.index(lon)

    r = _REASON_LINE.search(text)
    if r:
        reason = r.group("body")
    elif m:  # no reason header: whatever follows the meta line
        reason = text[m.end():]
    else:
        reason = ""
    reason = re.sub(r"\s+", " ", reason).strip().strip('"').strip("'").strip() or None

    errors = []
    if meta_idx is None:
        errors.append(f"meta-action: {err or 'not found'}")
    if reason is None:
        errors.append("reason: missing")
    return ParseResult(meta_idx, reason, ok=not errors, error="; ".join(errors) or None, partial=partial)
