"""A(통합) 핵심 로직 — 구간분할 _run_refine + 트리거 + adaLN 삽입 + 되먹임.

실제 motion_planning_head.py 에 옮겨 넣을 '통합 로직의 CPU 검증판'.
FakeRefineHead + MockLanguageModule + AdaLNZeroStub 로 GPU 없이 흐름을 검증한다.

[되먹임 설계 — 문서 기준: refine 층을 두 구간으로]
  앞 구간  : 층 [0:split]  (1차·2차 공통 → 1번만 실행하고 재사용 = 가벼움)
  1차      : 앞 + 층 [split:num] (cond=None) → plan_reg1, plan_cls1
  트리거   : 샘플마다 plan_cls1(불확실성)이 임계값을 넘으면
  LM       : [(meta_idx, sentence), ...] = lm.infer_batch(트리거된 샘플들)  # C 제공 인터페이스
  adaLN    : cond = adaln(meta_idx [B]) -> (gamma, beta, alpha)           # B 제공 인터페이스
  2차      : 앞(재사용) + 층 [split:num] (cond 주입, adaln.modulate) → plan_reg2
  트리거 + 언어 성공한 샘플만 2차 결과, 나머지(off·미발동·언어 실패)는 1차 결과 그대로

[A가 보장해야 하는 성질 (CPU에서 검증, test_integration.py)]
  1) use_feedback=False  →  원본(전체 1회)과 '완전히 동일'
  2) adaLN-Zero(0 초기화) →  트리거돼도 출력이 1차와 동일 (안전 시작)
  3) adaLN 가중치가 0이 아니면 → 2차 출력이 달라짐 (되먹임이 실제로 작동)
  4) 0 초기화에서도 adaLN에 기울기가 흐름 (학습 가능)
  5) 언어 실패(None) 샘플 / 트리거 안 된 샘플은 1차 결과 유지
"""

import torch
import torch.nn as nn


# =============================================================================
#  B·C의 인터페이스 (A는 '스텁/목'으로 먼저 만들어 통합 흐름을 검증한다)
# =============================================================================
class AdaLNZeroStub(nn.Module):
    """B가 제공할 AdaLNZero 의 초기/가짜 버전.

    인터페이스:  forward(meta_idx) -> (gamma, beta, alpha), 각 [B, 1, D] (int면 [1, 1, D])
                 modulate(h, gamma, beta, alpha) -> h + alpha * ((1 + gamma) * LN(h) + beta)
    0 초기화라 처음엔 항등(출력 안 바뀜). (1 + gamma)·LN(h) 덕분에 0에서도 alpha가 학습된다.
    """

    def __init__(self, embed_dims=256, num_meta=12):
        super().__init__()
        self.embed_dims = embed_dims
        self.emb = nn.Embedding(num_meta, embed_dims * 3)
        nn.init.zeros_(self.emb.weight)     # ★ 0 초기화 = 안전 시작
        self.norm = nn.LayerNorm(embed_dims, elementwise_affine=False, eps=1e-6)

    def forward(self, meta_idx):
        if not torch.is_tensor(meta_idx):
            meta_idx = torch.tensor([meta_idx])
        v = self.emb(meta_idx.long().reshape(-1))[:, None]   # [B, 1, 3D]
        gamma, beta, alpha = v.chunk(3, dim=-1)
        return gamma, beta, alpha

    def modulate(self, h, gamma, beta, alpha):
        """B AdaLNZero.modulate 와 같은 식"""
        return h + alpha * ((1 + gamma) * self.norm(h) + beta)


class MockLanguageModule:
    """C가 제공할 language_module 의 가짜 버전 (A 테스트용).

    인터페이스:  infer(scene_input) -> (meta_idx:int, sentence:str)
                 infer_batch([scene_input, ...]) -> [(meta_idx, sentence), ...]
    scene_input 이 {"fail": True} 면 (None, None) — 언어 모델 실패 흉내
    """

    def __init__(self, fixed_meta=5, sentence="앞차 간격 유지, 직진"):
        self.fixed_meta = fixed_meta
        self.sentence = sentence

    def infer(self, scene_input):
        if isinstance(scene_input, dict) and scene_input.get("fail"):
            return None, None
        return self.fixed_meta, self.sentence

    def infer_batch(self, scene_inputs):
        return [self.infer(x) for x in scene_inputs]


# =============================================================================
#  A 핵심 함수들  (실제로는 MotionPlanningHead의 메서드로 옮긴다)
# =============================================================================
def run_refine_segment(head, feat, start, end, cond=None, modulate=None):
    """refine 층 [start:end] 를 실행. cond가 있으면 각 층 norm 자리에 adaLN 주입."""
    for i in range(start, end):
        feat = head.layer_forward(i, feat, cond=cond, modulate=modulate)
    return feat


def trigger_uncertain(plan_cls, threshold):
    """샘플마다 불확실하면 True(언어 호출). plan_cls 신뢰도의 최댓값이 임계값 미만이면 불확실.
    plan_cls [B, N] -> [B] bool"""
    conf = plan_cls.sigmoid().flatten(1)
    return conf.max(dim=1).values < threshold


def select_by_sample(use, new, old):
    """use [B] 인 샘플은 new, 나머지는 old"""
    return torch.where(use.reshape([-1] + [1] * (old.dim() - 1)), new, old)


def feedback_refine(
    head,
    feat,
    *,
    use_feedback=False,
    split=3,
    lm=None,
    adaln=None,
    scene_inputs=None,
    threshold=0.5,
):
    """1차 → (샘플별 트리거) → 언어 → adaLN → 2차. off면 1차 그대로.

    scene_inputs: 샘플마다 하나씩, 길이 B 리스트 (None이면 [None] * B)
    Returns: reg, cls, info(dict: triggered(bool, 한 샘플이라도 2차 사용)/used([B] bool)/meta_idx/sentence)
    """
    num = head.num_layers
    info = {"triggered": False, "used": None, "meta_idx": None, "sentence": None}

    # 앞 구간은 1차·2차 공통 → 한 번만 실행하고 재사용 (2차를 가볍게)
    feat_front = run_refine_segment(head, feat, 0, split, cond=None)

    # 1차: 뒤 구간 (cond 없음)
    feat1 = run_refine_segment(head, feat_front, split, num, cond=None)
    reg1, cls1 = head.head(feat1)

    if not use_feedback:
        return reg1, cls1, info
    use = trigger_uncertain(cls1, threshold)  # [B]
    if not use.any():
        return reg1, cls1, info

    # 트리거된 샘플만 언어 호출 → 실패(None) 샘플은 1차 유지
    bs = feat.shape[0]
    if scene_inputs is None:
        scene_inputs = [None] * bs
    idx = use.nonzero().flatten().tolist()
    results = lm.infer_batch([scene_inputs[b] for b in idx])
    meta_idx = torch.zeros(bs, dtype=torch.long)
    sentence = [None] * bs
    for b, (m, s) in zip(idx, results):
        if m is None:
            use[b] = False
        else:
            meta_idx[b], sentence[b] = int(m), s
    if not use.any():
        return reg1, cls1, info

    # 조건 → 2차 (뒤 구간만 재실행 = 동적 연산), 사용 샘플만 2차 결과로 교체
    cond = adaln(meta_idx)  # 각 [B, 1, D]
    feat2 = run_refine_segment(head, feat_front, split, num, cond=cond, modulate=adaln.modulate)
    reg2, cls2 = head.head(feat2)
    reg = select_by_sample(use, reg2, reg1)
    cls = select_by_sample(use, cls2, cls1)
    info.update({"triggered": True, "used": use.tolist(), "meta_idx": meta_idx.tolist(), "sentence": sentence})
    return reg, cls, info
