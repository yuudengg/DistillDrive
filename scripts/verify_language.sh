#!/usr/bin/env bash
# =============================================================================
#  verify_language.sh  —  A(통합) → 경량화 팀 GPU 담당자에게 전달하는 1회 검증 패키지
# =============================================================================
#  목적: A는 GPU가 없으니, 실제 모델에서 확인할 3가지를 'GPU 담당자가 한 줄로' 실행하게 만든다.
#  사용법:  bash verify_language.sh <checkpoint_경로> [샘플수(기본 50)]
#  결과:    각 단계 PASS/FAIL 과 로그. 실패하면 로그 파일 전체를 A에게 전달 요청.
#
#  ※ 아래 python tools/*.py 는 실제 repo의 검증 엔트리포인트로 교체해야 함(A가 채움).
#    - test_regression:   언어모듈 OFF로 돌린 출력이 원본과 동일한지 (allclose)
#    - test_segment_equiv: _run_refine 전체 == (앞+뒤) 구간, cond=None 동등성
#    - dump_features:     통합·라벨 파이프라인 입력용 특징/궤적 덤프 생성
# =============================================================================
set -euo pipefail

CKPT="${1:?checkpoint 경로를 넣어주세요}"
N="${2:-50}"
LOG="verify_$(date +%Y%m%d_%H%M%S).log"

echo "=== 언어모듈 통합 검증 (ckpt=$CKPT, samples=$N) ===" | tee "$LOG"

echo "[1/3] 회귀: 언어모듈 OFF == 원본 (출력 동일해야 PASS)" | tee -a "$LOG"
python tools/test_regression.py --ckpt "$CKPT" --num "$N" --lang off 2>&1 | tee -a "$LOG"

echo "[2/3] 구간분할 동등성: _run_refine 전체 == (앞 0:split)+(뒤 split:N), cond=None" | tee -a "$LOG"
python tools/test_segment_equiv.py --ckpt "$CKPT" --num "$N" 2>&1 | tee -a "$LOG"

echo "[3/3] 덤프 생성: 통합·라벨 파이프라인 입력용" | tee -a "$LOG"
python tools/dump_features.py --ckpt "$CKPT" --num "$N" --out dumps/ 2>&1 | tee -a "$LOG"

echo "=== 완료. '$LOG' 를 A에게 전달해주세요. 실패 시 로그 전체 첨부 부탁드립니다. ===" | tee -a "$LOG"
