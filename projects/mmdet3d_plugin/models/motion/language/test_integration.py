"""A 통합 로직 CPU 검증 테스트.

실행:  python test_integration.py
(torch만 있으면 됨. mmcv·실제 모델 불필요)

검증하는 성질 — A가 GPU에 넘기기 전에 CPU에서 반드시 통과시켜야 함:
  1) use_feedback=False  →  원본(전체 1회)과 완전히 동일
  2) adaLN-Zero(0 초기화) →  트리거돼도 출력이 1차와 동일 (안전 시작)
  3) adaLN 가중치가 0이 아니면 → 2차 출력이 달라짐 (되먹임 작동)
  4) 0 초기화에서도 adaLN에 기울기가 흐름 (학습 가능)
  5) 언어 실패(None) 샘플 / 트리거 안 된 샘플은 1차 결과 유지
feature는 실제 plan query처럼 [B, N, D] (N = 18).
"""

import torch

from fake_refine_head import FakeRefineHead
from feedback_integration import (
    AdaLNZeroStub,
    MockLanguageModule,
    feedback_refine,
    run_refine_segment,
    trigger_uncertain,
)


def _make(seed=0, num_layers=6, embed_dims=32, bs=2, num_modes=18):
    head = FakeRefineHead(embed_dims=embed_dims, num_layers=num_layers, seed=seed)
    head.eval()
    feat = torch.randn(bs, num_modes, embed_dims)  # 실제 plan query처럼 [B, N, D]
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
        scene_inputs=None, threshold=1.0,
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
        scene_inputs=None, threshold=1.0,
    )
    base_reg, _ = head.run_whole(feat)
    assert info["triggered"] is True
    assert not torch.allclose(reg_fb, base_reg, atol=1e-4), "되먹임이 출력을 바꿔야 함"


def test_zero_init_learns():
    """성질4: 0 초기화에서도 adaLN에 기울기가 흐름 (전부 0이면 영원히 학습 안 되는 문제 방지)."""
    head, feat = _make(bs=4)
    adaln = AdaLNZeroStub(embed_dims=head.embed_dims, num_meta=12)
    reg, _, info = feedback_refine(
        head, feat, use_feedback=True, split=3, lm=MockLanguageModule(fixed_meta=5),
        adaln=adaln, threshold=1.0,
    )
    reg.pow(2).mean().backward()
    assert info["triggered"] is True
    assert adaln.emb.weight.grad.abs().sum() > 0, "0 초기화에서 기울기가 0이면 학습 안 됨"


def test_none_keeps_first_pass():
    """성질5-1: 언어 실패(None) 샘플만 1차 결과 유지, 나머지는 2차 결과."""
    head, feat = _make()
    adaln = AdaLNZeroStub(embed_dims=head.embed_dims, num_meta=12)
    torch.nn.init.normal_(adaln.emb.weight, std=0.1)
    reg_fb, _, info = feedback_refine(
        head, feat, use_feedback=True, split=3, lm=MockLanguageModule(fixed_meta=5), adaln=adaln,
        scene_inputs=[{"fail": True}, None], threshold=1.0,
    )
    base_reg, _ = head.run_whole(feat)
    assert info["used"] == [False, True]
    assert torch.allclose(reg_fb[0], base_reg[0], atol=1e-6), "언어 실패 샘플은 1차 그대로"
    assert not torch.allclose(reg_fb[1], base_reg[1], atol=1e-4), "성공 샘플은 2차 결과"
    _, _, info = feedback_refine(
        head, feat, use_feedback=True, split=3, lm=MockLanguageModule(), adaln=adaln,
        scene_inputs=[{"fail": True}] * 2, threshold=1.0,
    )
    assert info["triggered"] is False, "전부 실패하면 2차 안 함"


def test_trigger_per_sample():
    """성질5-2: 트리거는 샘플마다 판단. 확실한 샘플은 언어 호출·2차 없음."""
    cls = torch.full((3, 18), -5.0)   # 모두 불확실 (sigmoid ~ 0.007)
    cls[1, 4] = 5.0                   # 샘플 1만 확실한 경로 하나 (sigmoid ~ 0.993)
    assert trigger_uncertain(cls, 0.5).tolist() == [True, False, True]


if __name__ == "__main__":
    tests = [
        ("성질1 off==원본", test_off_equals_original),
        ("보조  구간분할 동등", test_segment_split_equiv),
        ("성질2 adaLN-Zero 안전", test_adaln_zero_is_safe),
        ("성질3 비영 adaLN 작동", test_nonzero_adaln_changes_output),
        ("성질4 0 초기화 학습 가능", test_zero_init_learns),
        ("성질5 언어 실패 샘플 1차 유지", test_none_keeps_first_pass),
        ("성질5 샘플별 트리거", test_trigger_per_sample),
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
