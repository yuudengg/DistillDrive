"""A(통합) 핵심 로직 — 구간분할 _run_refine + 트리거 + adaLN 삽입 + 되먹임.

실제 motion_planning_head.py 에 옮겨 넣을 '통합 로직의 CPU 검증판'.
FakeRefineHead + MockLanguageModule + AdaLNZeroStub 로 GPU 없이 흐름을 검증한다.

[되먹임 설계 — 문서 기준: refine 층을 두 구간으로]
  앞 구간  : 층 [0:split]  (1차·2차 공통 → 1번만 실행하고 재사용 = 가벼움)
  1차      : 앞 + 층 [split:num] (cond=None) → plan_reg1, plan_cls1
  트리거   : plan_cls1(불확실성)이 임계값을 넘으면
  LM       : meta_idx, sentence = lm.infer(scene_input)      # C 제공 인터페이스
  adaLN    : cond = adaln(meta_idx) -> (gamma, beta, alpha)  # B 제공 인터페이스
  2차      : 앞(재사용) + 층 [split:num] (cond 주입) → plan_reg2
  off 또는 트리거 미발동 → 1차 결과 그대로

[A가 보장해야 하는 성질 (CPU에서 검증, test_integration.py)]
  1) use_feedback=False  →  원본(전체 1회)과 '완전히 동일'
  2) adaLN-Zero(0 초기화) →  트리거돼도 출력이 1차와 동일 (안전 시작)
  3) adaLN 가중치가 0이 아니면 → 2차 출력이 달라짐 (되먹임이 실제로 작동)
"""

import torch
import torch.nn as nn


# =============================================================================
#  B·C의 인터페이스 (A는 '스텁/목'으로 먼저 만들어 통합 흐름을 검증한다)
# =============================================================================
class AdaLNZeroStub(nn.Module):
    """B가 제공할 AdaLNZero 의 초기/가짜 버전.

    인터페이스:  forward(meta_idx) -> (gamma, beta, alpha)
    0 초기화라 처음엔 항등(출력 안 바뀜). 학습하며 서서히 작동(adaLN-Zero).
    """

    def __init__(self, embed_dims=256, num_meta=12):
        super().__init__()
        self.embed_dims = embed_dims
        self.emb = nn.Embedding(num_meta, embed_dims * 3)
        nn.init.zeros_(self.emb.weight)     # ★ 0 초기화 = 안전 시작

    def forward(self, meta_idx):
        if not torch.is_tensor(meta_idx):
            meta_idx = torch.tensor([meta_idx])
        v = self.emb(meta_idx.long())        # [B, 3D]
        gamma, beta, alpha = v.chunk(3, dim=-1)
        return gamma, beta, alpha


class MockLanguageModule:
    """C가 제공할 language_module 의 가짜 버전 (A 테스트용).

    인터페이스:  infer(scene_input) -> (meta_idx:int, sentence:str)
    """

    def __init__(self, fixed_meta=5, sentence="앞차 간격 유지, 직진"):
        self.fixed_meta = fixed_meta
        self.sentence = sentence

    def infer(self, scene_input):
        return self.fixed_meta, self.sentence


# =============================================================================
#  A 핵심 함수들  (실제로는 MotionPlanningHead의 메서드로 옮긴다)
# =============================================================================
def run_refine_segment(head, feat, start, end, cond=None):
    """refine 층 [start:end] 를 실행. cond가 있으면 각 층 norm 자리에 adaLN 주입."""
    for i in range(start, end):
        feat = head.layer_forward(i, feat, cond=cond)
    return feat


def trigger_uncertain(plan_cls, threshold):
    """불확실하면 True(언어 호출). plan_cls 신뢰도의 최댓값이 임계값 미만이면 불확실."""
    conf = plan_cls.sigmoid()
    return bool((conf.max() < threshold).item())


def feedback_refine(
    head,
    feat,
    *,
    use_feedback=False,
    split=3,
    lm=None,
    adaln=None,
    scene_input=None,
    threshold=0.5,
):
    """1차 → (트리거) → 언어 → adaLN → 2차. off면 1차 그대로.

    Returns: reg, cls, info(dict: triggered/meta_idx/sentence)
    """
    num = head.num_layers
    info = {"triggered": False, "meta_idx": None, "sentence": None}

    # 앞 구간은 1차·2차 공통 → 한 번만 실행하고 재사용 (2차를 가볍게)
    feat_front = run_refine_segment(head, feat, 0, split, cond=None)

    # 1차: 뒤 구간 (cond 없음)
    feat1 = run_refine_segment(head, feat_front, split, num, cond=None)
    reg1, cls1 = head.head(feat1)

    if not use_feedback:
        return reg1, cls1, info
    if not trigger_uncertain(cls1, threshold):
        return reg1, cls1, info

    # 트리거 발동 → 언어 → 조건 → 2차 (뒤 구간만 재실행 = 동적 연산)
    meta_idx, sentence = lm.infer(scene_input)
    cond = adaln(meta_idx)
    feat2 = run_refine_segment(head, feat_front, split, num, cond=cond)
    reg2, cls2 = head.head(feat2)
    info.update({"triggered": True, "meta_idx": meta_idx, "sentence": sentence})
    return reg2, cls2, info
