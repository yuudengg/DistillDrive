"""A 통합 로직 CPU 검증 테스트.

실행:  python test_integration.py
(torch만 있으면 됨. mmcv·실제 모델 불필요)

검증하는 3가지 성질 — A가 GPU에 넘기기 전에 CPU에서 반드시 통과시켜야 함:
  1) use_feedback=False  →  원본(전체 1회)과 완전히 동일
  2) adaLN-Zero(0 초기화) →  트리거돼도 출력이 1차와 동일 (안전 시작)
  3) adaLN 가중치가 0이 아니면 → 2차 출력이 달라짐 (되먹임 작동)
"""

import torch

from fake_refine_head import FakeRefineHead
from feedback_integration import (
    AdaLNZeroStub,
    MockLanguageModule,
    feedback_refine,
    run_refine_segment,
)


def _make(seed=0, num_layers=6, embed_dims=32):
    head = FakeRefineHead(embed_dims=embed_dims, num_layers=num_layers, seed=seed)
    head.eval()
    feat = torch.randn(2, embed_dims)
    return head, feat


def test_off_equals_original():
    """성질1: 되먹임 off == 전체 1회 실행(원본)."""
    head, feat = _make()
    base_reg, base_cls = head.run_whole(feat)
    reg, cls, info = feedback_refine(head, feat, use_feedback=False)
    assert torch.allclose(reg, base_reg, atol=1e-6)
    assert torch.allclose(cls, base_cls, atol=1e-6)
    assert info["triggered"] is False


def test_segment_split_equiv():
    """보조: 전체 [0:N] == [0:split] + [split:N] (cond=None)."""
    head, feat = _make()
    whole = run_refine_segment(head, feat, 0, head.num_layers, cond=None)
    split = 3
    part = run_refine_segment(head, feat, 0, split, cond=None)
    part = run_refine_segment(head, part, split, head.num_layers, cond=None)
    assert torch.allclose(whole, part, atol=1e-6)


def test_adaln_zero_is_safe():
    """성질2: adaLN-Zero(0 초기화)면 트리거돼도 출력이 1차와 동일."""
    head, feat = _make()
    adaln = AdaLNZeroStub(embed_dims=head.embed_dims, num_meta=12)  # 0 초기화
    lm = MockLanguageModule(fixed_meta=5)
    # 임계값 1.0 → 항상 트리거 (sigmoid<1)
    reg_fb, cls_fb, info = feedback_refine(
        head, feat, use_feedback=True, split=3, lm=lm, adaln=adaln,
        scene_input=None, threshold=1.0,
    )
    base_reg, base_cls = head.run_whole(feat)
    assert info["triggered"] is True
    assert torch.allclose(reg_fb, base_reg, atol=1e-6), "0-init인데 출력이 바뀌면 안 됨"


def test_nonzero_adaln_changes_output():
    """성질3: adaLN 가중치가 0이 아니면 2차 출력이 달라져야 함(되먹임 작동)."""
    head, feat = _make()
    adaln = AdaLNZeroStub(embed_dims=head.embed_dims, num_meta=12)
    torch.nn.init.normal_(adaln.emb.weight, std=0.1)  # 학습됐다고 가정: 0이 아님
    lm = MockLanguageModule(fixed_meta=5)
    reg_fb, _, info = feedback_refine(
        head, feat, use_feedback=True, split=3, lm=lm, adaln=adaln,
        scene_input=None, threshold=1.0,
    )
    base_reg, _ = head.run_whole(feat)
    assert info["triggered"] is True
    assert not torch.allclose(reg_fb, base_reg, atol=1e-4), "되먹임이 출력을 바꿔야 함"


if __name__ == "__main__":
    tests = [
        ("성질1 off==원본", test_off_equals_original),
        ("보조  구간분할 동등", test_segment_split_equiv),
        ("성질2 adaLN-Zero 안전", test_adaln_zero_is_safe),
        ("성질3 비영 adaLN 작동", test_nonzero_adaln_changes_output),
    ]
    ok = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            ok += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
    print(f"\n{ok}/{len(tests)} passed")
