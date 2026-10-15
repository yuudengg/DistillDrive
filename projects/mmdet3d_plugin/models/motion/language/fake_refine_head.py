"""FakeRefineHead — A(통합)가 B·C에게 제공하는 '가짜 정제 헤드'.

목적: 실제 모델(DistillDrive) 없이 CPU에서 '되먹임 통합 로직'을 검증하기 위한 스탠드인.
      실제 motion_planning_head의 refine 디코더와 '같은 구조'(여러 층을 순서대로 실행)를
      흉내내되, 내부는 단순 nn.Linear 라 가볍고 빠르다.

검증 용도:
  - "전체 1회 실행  ==  두 구간(앞 split, 뒤 split)으로 나눠 실행" 동등성
  - adaLN 조건(cond)을 각 층의 norm 자리에 삽입했을 때의 흐름
  - adaLN-Zero(0 초기화)면 조건을 넣어도 출력이 안 바뀌는 '안전 시작' 성질

주의: 실제 수치 성능과는 무관한 '로직 검증용'이다. 실제 모델 검증은 GPU 팀에 요청한다.
"""

import torch
import torch.nn as nn


class FakeRefineHead(nn.Module):
    def __init__(self, embed_dims=256, num_layers=6, fut_ts=6, out_dim=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)          # 재현성: 동등성 테스트에 필수
        self.embed_dims = embed_dims
        self.num_layers = num_layers     # 실제 refine 디코더 층 수(예: 6)를 흉내
        self.fut_ts = fut_ts
        self.out_dim = out_dim
        # 각 '층' = norm/ffn 역할의 Linear (실제 operation_order의 한 블록 대용)
        self.layers = nn.ModuleList(
            [nn.Linear(embed_dims, embed_dims) for _ in range(num_layers)]
        )
        self.act = nn.ReLU()
        # 마지막에 궤적(plan_reg)과 점수(plan_cls=불확실성 신호)를 뽑는 head
        self.reg_head = nn.Linear(embed_dims, fut_ts * out_dim)
        self.cls_head = nn.Linear(embed_dims, 1)

    def layer_forward(self, i, feat, cond=None, modulate=None):
        """한 '층'을 실행. cond(=adaLN 조건)가 있으면 norm 자리에 adaLN-Zero 주입.

        feat: [B, N, D] (실제 plan query처럼 경로 N개)
        cond = (gamma, beta, alpha), 각 [B, 1, D]
          gamma: scale, beta: shift, alpha: gate(0이면 항등 → 원본 그대로)
        modulate: B의 AdaLNZero.modulate (식은 B 파일 한 곳에만 둔다)
        """
        h = self.act(self.layers[i](feat))
        if cond is not None:
            h = modulate(h, *cond)                  # h + alpha * ((1 + gamma) * LN(h) + beta)
        return h

    def head(self, feat):
        bs, n = feat.shape[:2]
        reg = self.reg_head(feat).reshape(bs, n, self.fut_ts, self.out_dim)  # [B, N, T, 2]
        cls = self.cls_head(feat).squeeze(-1)       # [B, N] 불확실성/신뢰 신호
        return reg, cls

    def run_whole(self, feat):
        """참고용: 전체 층을 한 번에 실행(= 분할 안 한 원본). 동등성 테스트 기준."""
        for i in range(self.num_layers):
            feat = self.layer_forward(i, feat, cond=None)
        return self.head(feat)
