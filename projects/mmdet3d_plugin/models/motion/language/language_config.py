# =============================================================================
#  언어 모듈 config 블록 (A가 관리)  —  실제 config 파일에 병합해서 사용
# =============================================================================
#  철칙: enabled=False(기본 off)일 때, 모델 출력은 원본과 '완전히 동일'해야 한다.
#        (회귀 테스트 통과 = 이 보장. test_integration.py의 성질1)
#  켜는 건 통합·검증이 끝난 뒤에만.
# =============================================================================

language_module = dict(
    enabled=False,            # ★ 기본 off. 통합/검증 끝나기 전엔 절대 True 금지.

    # --- 구간 분할(⑤): refine 디코더 층을 앞/뒤로 나누는 지점 ---
    #     1차 = 층[0:split] + 층[split:num],  2차 = 앞 재사용 + 층[split:num](cond)
    num_refine_layers=6,      # 실제 refine 디코더 층 수 (config에서 확인 후 맞추기)
    split=3,                  # 분할 지점 (앞 0~3, 뒤 3~6)

    # --- 트리거(⑥): 언제 언어를 부를지 ---
    trigger_threshold=0.5,    # plan_cls 신뢰도 최댓값 < 이 값이면 '불확실' → 언어 호출
    #  (매 프레임 X. 불확실할 때만 → 동적 연산 = 노벨티)

    # --- adaLN(③, B 제공): 조건 → (gamma, beta, alpha), 0 초기화 ---
    adaln=dict(
        type='AdaLNZero',     # B가 구현. 인터페이스: forward(meta_idx)->(g,b,a)
        embed_dims=256,
        num_meta=12,          # 메타행동 12지
    ),

    # --- 언어 모델(④, C 제공) ---
    language_model=dict(
        type='LanguageModule',  # C가 구현. 인터페이스: infer(scene_input)->(meta_idx, sentence)
        # (모델·체크포인트 경로 등은 C가 채움)
    ),

    # --- 안전장치(⑧, B 제공, 선택) ---
    counterfactual_check=False,  # 2차 궤적이 1차보다 나은지 확인, 나쁘면 롤백 (추후 on)
    rollback_mode="pool",        # B(safety_check.py): "pool" = 1차+2차 후보 12개를 충돌 검사 후 최고 점수
                                 #                     "collision" = 2차가 충돌이고 1차는 안전할 때만 1차로
)
