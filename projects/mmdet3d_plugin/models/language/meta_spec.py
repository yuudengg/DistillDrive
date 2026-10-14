"""Meta-action numbering = the ONE place to change if the team changes the convention.

Convention (팀 확정 2026-10-13, A 인터페이스 스펙 v2 = 데이터셋·앵커·체크포인트·meta_action.py 기준):
    meta_idx = lateral * 4 + longitudinal          (0 ~ 11)
    lateral      : 0 right(우) / 1 left(좌) / 2 straight(직)
    longitudinal : 0 accelerate(가속) / 1 keep(유지) / 2 decelerate(감속) / 3 stop(정지)

Canonical text name = "<lateral>+<longitudinal>", e.g. 6 -> "left+decelerate".
The label files written by C and the LM output both use this name (not the number).

from_rls_index / to_rls_index are kept for old callers; with the unified order they are identity.
"""
from typing import Tuple

LATERAL = ("right", "left", "straight")  # same order as models/motion/meta_action.py
LONGITUDINAL = ("accelerate", "keep", "decelerate", "stop")
NUM_LATERAL = len(LATERAL)
NUM_LONGITUDINAL = len(LONGITUDINAL)
NUM_META = NUM_LATERAL * NUM_LONGITUDINAL  # 12

LATERAL_KO = {"left": "좌회전", "straight": "직진", "right": "우회전"}
LONGITUDINAL_KO = {"accelerate": "가속", "keep": "유지", "decelerate": "감속", "stop": "정지"}


def meta_index(lateral: str, longitudinal: str) -> int:
    """('left', 'decelerate') -> 6"""
    return LATERAL.index(lateral) * NUM_LONGITUDINAL + LONGITUDINAL.index(longitudinal)


def split_meta(meta_idx: int) -> Tuple[int, int]:
    """6 -> (1, 2)  (lateral index, longitudinal index)"""
    meta_idx = int(meta_idx)
    if not 0 <= meta_idx < NUM_META:
        raise ValueError(f"meta_idx must be in [0, {NUM_META}), got {meta_idx}")
    return meta_idx // NUM_LONGITUDINAL, meta_idx % NUM_LONGITUDINAL


def meta_name(meta_idx: int) -> str:
    """6 -> 'left+decelerate'"""
    lat, lon = split_meta(meta_idx)
    return f"{LATERAL[lat]}+{LONGITUDINAL[lon]}"


def meta_name_ko(meta_idx: int) -> str:
    """6 -> '좌회전+감속'"""
    lat, lon = split_meta(meta_idx)
    return f"{LATERAL_KO[LATERAL[lat]]}+{LONGITUDINAL_KO[LONGITUDINAL[lon]]}"


# ----------------------------------------------------------------------------------
# meta_action.py numbering (identical now that the order is unified; kept so old callers still work)
# ----------------------------------------------------------------------------------
_RLS_TO_DOC_LATERAL = {0: LATERAL.index("right"), 1: LATERAL.index("left"), 2: LATERAL.index("straight")}


def from_rls_index(meta_idx_rls: int) -> int:
    """meta_action.py index -> this convention (identity since the 2026-10-13 order decision)."""
    meta_idx_rls = int(meta_idx_rls)
    if not 0 <= meta_idx_rls < NUM_META:
        raise ValueError(f"meta_idx must be in [0, {NUM_META}), got {meta_idx_rls}")
    lat, lon = divmod(meta_idx_rls, NUM_LONGITUDINAL)
    return _RLS_TO_DOC_LATERAL[lat] * NUM_LONGITUDINAL + lon


def to_rls_index(meta_idx: int) -> int:
    """Inverse of from_rls_index."""
    lat, lon = split_meta(meta_idx)
    inv = {v: k for k, v in _RLS_TO_DOC_LATERAL.items()}
    return inv[lat] * NUM_LONGITUDINAL + lon
