# DistillDrive 언어 모듈 (STEP 3: ②④⑦) 구조와 사용법

기준: language 브랜치 + `refine B안(_refine_once)` + `MetaActionHead` (2026-10-04 보고서 기준)

## 1. 한 줄 설계

주행 모델이 만든 **언어 패킷**(ego·주변 객체·지도 토큰 약 20개 + 선택된 궤적 + 메타행동)을 디스크에 저장해 두고,
언어팀은 그 패킷만으로 **Qwen2.5 + 투영기(projector)** 를 학습해 "왜 그렇게 가는지" 한 문장을 만든다.
주행 모델은 건드리지 않으므로 기존 학습·평가 결과는 그대로다.

```
[주행 모델 forward]                                   [언어팀 CPU/GPU]
 refine → 1차 궤적(plan_reg, plan_cls)                 packets/<token>.pt  +  reasons.jsonl
   │                                                        │
   ├─ MetaActionHead (규칙) ─→ meta-action(0~11)             ▼
   ├─ 토큰 선택: ego(1) + agent(K) + map(M)           ReasonPacketDataset
   └─ LangPacketBuilder ─→ lang_packet ──저장──►      LanguageHead
                                                       ├ SceneProjector (256 → LM dim)
 (되먹임 adaLN 가지는 별도, 같은 meta-action 사용)        ├ Qwen2.5-1.5B (동결, 단계 2에서 LoRA)
                                                       └ 입력 = [프롬프트 | 장면 토큰 | 메타행동 문장 | Reason:] → 근거 문장
 라벨(오프라인): Qwen2.5-VL(카메라 영상 + GT 메타행동) → reasons.jsonl   ← 학습용 모델과 다른 모델
```

## 2. 왜 이렇게 나눴나

| 결정 | 이유 |
| --- | --- |
| 패킷을 디스크에 캐시 | 언어팀은 GPU를 직접 못 씀. GPU 담당이 패킷을 **한 번만** 뽑아 주면 이후 작업은 CPU 디버깅 + 짧은 GPU 학습으로 가능 |
| 주행 모델 forward 안에서 LM 호출 안 함 | 3B 모델이 평가 속도·메모리·재현성에 영향을 주지 않게 하려고. `lang_packet_cfg=None`이면 코드 경로 자체가 꺼짐 |
| 패킷은 detach | 언어 loss가 궤적 성능을 건드리지 않음 (나중에 end-to-end가 필요하면 그때 열면 됨) |
| 장면 토큰은 약 20개 | SparseDrive의 희소 인스턴스 표현을 그대로 활용. BEV 전체를 넣는 것보다 가볍고 빠름 |
| 메타행동은 규칙(MetaActionHead) | 학습 파라미터 0. 프롬프트에 **텍스트로** 넣어 줘서 LM이 "무슨 행동의 이유인지" 알고 이유만 생성 |
| 라벨 모델 ≠ 학습 모델 | Qwen2.5-VL이 영상을 보고 라벨 작성 → Qwen2.5-1.5B가 장면 토큰만 보고 같은 근거를 재현하도록 학습 (지식 증류와 같은 구도) |
| 2단계 학습 | 1단계: 투영기만 (LM 동결) → 2단계: 투영기 + LoRA. 1단계만으로도 파이프라인이 도는지 먼저 확인 가능 |

## 3. 파일 목록

```
projects/mmdet3d_plugin/models/language/        (새 폴더)
  prompt.py         메타행동 ↔ 문장, 프롬프트, 라벨 프롬프트, 라벨 검증·일관성 지표 (torch 불필요)
  packet.py         LangPacketBuilder(주행 모델 forward 안에서 실행), 저장/로드, PacketDumper(forward hook)
  projector.py      SceneProjector: 장면 토큰 + 궤적 토큰 → LM 임베딩
  language_head.py  LanguageHead: 입력 조립, loss(근거 토큰만), generate, save/load, LoRA
  dataset.py        ReasonPacketDataset, collate_fn
  tiny.py           CPU 디버그용 작은 랜덤 Qwen2 + 토크나이저 (다운로드 없음)
  __init__.py
tools/language/
  _compat.py              mmcv 없이도 실제 meta_action.py를 불러오는 도우미
  make_meta_action_gt.py  GT 궤적 → 12클래스 메타행동 + 분포 (README 확인 #2)  ← CPU
  gen_reason_labels.py    Qwen2.5-VL로 근거 라벨 생성                           ← GPU
  train_language.py       단계 1/2 학습                                          ← GPU (--tiny는 CPU)
  eval_language.py        생성 품질 + 1차 궤적 메타행동 정확도                    ← GPU
tests/language/test_language_pipeline.py   CPU 스모크 테스트
patches/apply_language_patch.py            motion_planning_head.py에 25줄 추가 (줄바꿈 CRLF/LF 그대로 유지, 권장)
patches/language_packet_hook.patch         같은 변경의 diff (참고용)
```

## 4. 적용 순서

필요 패키지: `torch`(패킷 덤프는 1.x에서도 동작, 언어모델 학습은 2.x 기준으로 확인), `transformers`(Qwen2.5-VL은 4.49 이상), `peft`(단계 2 LoRA), `accelerate`(device_map 사용 시), 라벨 생성에는 `pillow`.

### 0) 파일 넣기 + 주행 모델 쪽 연결
```bash
cp -r projects tools tests patches <저장소 맨 위>/      # language 폴더, tools/language, tests/language, patches
cd <저장소 맨 위>
python patches/apply_language_patch.py --check   # 5군데 위치가 모두 찾아지는지만 확인
python patches/apply_language_patch.py           # 수정 (원본은 motion_planning_head.py.bak 으로 백업)
```
원본 `motion_planning_head.py`는 Windows 줄바꿈(CRLF)이라 `git apply`가 잘 안 먹을 수 있어서, 줄바꿈을 그대로 유지하는 스크립트를 권장한다.
스크립트는 (1) import 1줄 (2) 생성자 인자 2개 (3) 빌더 생성 4줄 (4) forward 끝부분 패킷 생성 블록 (5) planning_output 한 줄을 더한다.
위치를 못 찾으면 그 파일이 서로 다르게 고쳐진 것이니 덮어쓰지 말고 같이 맞추자. 이미 적용된 파일이면 아무것도 하지 않는다.

config에 추가 (끄려면 이 줄을 빼면 됨):
```python
lang_packet_cfg=dict(type="LangPacketBuilder", num_agents=12, num_map=6, ego_fut_ts=6, ego_fut_mode=6,
                     meta_action=dict(dt=0.5, lat_thresh=2.0)),
lang_packet_refine_idx=-1,   # 되먹임 연결 후에는 2 (= refine 3 직후)
```

### 1) CPU: 테스트 + GT 메타행동 + 임계값 분포 (언어팀 노트북)
```bash
pytest tests/language -q
python tools/language/make_meta_action_gt.py --pkl data/infos/nuscenes_infos_train.pkl --out data/language/meta_action_gt_train.json
python tools/language/make_meta_action_gt.py --pkl data/infos/nuscenes_infos_val.pkl   --out data/language/meta_action_gt_val.json
```
테스트는 저장소 안의 `projects/mmdet3d_plugin/models/motion/meta_action.py`(팀원 코드)를 불러오므로 저장소 맨 위에서 실행해야 한다.
출력의 `lateral class == gt_ego_fut_cmd` 가 100% 근처인지, 12클래스 분포가 한쪽으로 쏠리지 않는지 확인 → 임계값(2.0m, ±20%, 0.5m/s) 조정.

### 2) GPU 담당: 패킷 추출 (체크포인트 1개로 train/val 각각 1회)
`tools/test.py`에서 체크포인트를 불러온 직후에 두 줄만 추가하고 평소처럼 평가를 돌리면 된다.
```python
from projects.mmdet3d_plugin.models.language import attach_packet_dumper
attach_packet_dumper(model, "data/language/packets_train")   # val은 packets_val
```
첫 실행 때 `packets_train/` 파일 이름이 nuScenes sample token으로 저장됐는지 꼭 확인 (아니면 `idx0000000` 형태로 저장됨 → `extract_sample_ids` 수정).

### 3) GPU: 근거 라벨 생성 (처음엔 --limit 16 으로 눈으로 확인)
```bash
python tools/language/gen_reason_labels.py --pkl data/infos/nuscenes_infos_train.pkl \
   --meta-json data/language/meta_action_gt_train.json --data-root data/nuscenes \
   --out data/language/reasons_train.jsonl --stride 4 --batch-size 8
```

### 4) 학습 → 평가
```bash
python tools/language/train_language.py --stage 1 --lm Qwen/Qwen2.5-1.5B-Instruct \
   --labels data/language/reasons_train.jsonl --packets data/language/packets_train \
   --val-labels data/language/reasons_val.jsonl --val-packets data/language/packets_val \
   --out work_dirs/language/stage1 --epochs 3 --batch-size 8 --lr 1e-3
python tools/language/train_language.py --stage 2 --init work_dirs/language/stage1/best --lr 2e-4 \
   --labels ... --packets ... --val-labels ... --val-packets ... --out work_dirs/language/stage2
python tools/language/eval_language.py --ckpt work_dirs/language/stage2/best --stage 2 \
   --labels data/language/reasons_val.jsonl --packets data/language/packets_val --meta-source pred \
   --out work_dirs/language/stage2/eval_pred.jsonl
```
CPU 디버그: 위 명령에 `--tiny --workers 0` 을 붙이면 다운로드 없이 작은 랜덤 모델로 전체 경로가 돈다.

## 5. 코드를 읽고 가정한 것 (확인 필요)

1. **`gt_ego_fut_trajs`는 스텝별 증분(delta)** (plan_reg와 같은 형태) → `make_meta_action_gt.py --traj-mode delta` 기본값. 위치(position)로 저장돼 있으면 `--traj-mode position`. 스크립트가 `gt_ego_fut_cmd`와의 일치율로 스스로 점검해 준다.
2. **좌표계: x = 오른쪽, y = 앞** (MetaActionHead 주석의 "final x >= 2m → right"와 일치). 사전조사 PDF의 "(X,Y)=(전방, 좌측)" 표기와 다르니 한 번 확인. `describe_objects`(옵션)가 이 가정을 쓴다.
3. **`metas["gt_ego_fut_cmd"]`** 가 forward에 들어오는 metas에 있다고 가정. 없으면 가장 점수 높은 모드를 고르는 경로로 자동 대체된다 (패킷의 `selected_mode`가 달라질 수 있으니 확인).
4. **ego 토큰** = 선택된 모드의 `plan_query`(refine 입력 쿼리). 패킷에는 refine 이후의 ego 상태가 따로 없어서 이 값을 썼다.
5. **agent 토큰** = 모든 refine 이후 최종 `instance_feature[:, :N]`. 되먹임 2차 패스가 생기면 어느 시점 값을 쓸지 `lang_packet_refine_idx`와 함께 정하면 된다.
6. 패킷 추출은 **eval 모드에서만** 만들어진다 (`not self.training`). 학습 중 패킷이 필요하면 패치의 조건을 바꿔야 한다.
7. 통합 계획 문서의 ②④⑦ 정의는 보고서에 직접 나와 있지 않아 **④ = 문장 생성(LanguageHead), ⑦ = 근거 라벨(gen_reason_labels), ② = 장면 토큰 구성(LangPacketBuilder + SceneProjector)** 로 해석했다. 다르면 알려 주세요.

## 6. 다음에 붙일 것 (이번 범위 밖)

- 되먹임(③⑥) 연결: `LanguageHead`는 되먹임과 독립이다. 나중에 근거 임베딩을 cond에 합치고 싶다면 `LanguageHead._build`의 출력 마지막 hidden state를 꺼내는 함수 하나만 더 추가하면 된다.
- 트리거(STEP 4): 위험 상황일 때만 언어를 생성하도록 `generate` 호출 조건만 바깥에서 걸면 된다.
- 근거가 장면과 맞는지 사람이 보는 소규모 검수(예: 100개)를 `eval_language.py`의 jsonl로 하는 것을 권장.

## 7. C 역할 인터페이스 (역할분담 문서 4장 #3)

```python
from projects.mmdet3d_plugin.models.language import build_language_module
lm = build_language_module(cfg)            # cfg=None → 꺼짐(기본값)
meta_idx, reason = lm.infer(scene_input)   # meta_idx: 0~11 int (문서 번호 체계, meta_spec.py) / 파싱 실패 시 (None, None)
```
- A·B용 mock: `dict(type="mock", mode="fixed"|"cycle"|"echo", fail_every=N, delay_s=...)`
- 실제 모델: `dict(type="hf", ckpt_dir=..., lm_path="Qwen/Qwen2.5-1.5B-Instruct", load_in_4bit=True)`
  현재는 임베딩 경로(scene/scene_mask/scene_type/traj, 배치 차원 없이)만 지원. 텍스트 요약 경로는 회의 결정 후 추가.
- 번호 체계: 라벨·패킷은 meta_action.py 순서(우/좌/직), LM 입출력은 문서 순서(좌/직/우). 변환은 `from_rls_index`/`to_rls_index` 한 곳에서만 한다.

학습·평가 플래그:
- `--predict-action`: LM이 `Meta-action: ...\nReason: ...`을 직접 쓰도록 학습 (hf 모듈은 이렇게 학습한 체크포인트만 받음). train과 eval에 둘 다 붙인다.
- `--qlora`: stage 2에서 4bit(nf4) 기반 LoRA. CUDA와 bitsandbytes가 필요하고, CPU에서는 검증하지 않았다.
- `gen_reason_labels.py --mock`: VLM과 이미지 없이 라벨 파이프라인을 CPU에서 확인.

## 8. 환경 (2026-10-08 결정: 원본 문서 환경, RTX 3090)

| 용도 | Python | torch | 그 외 | 확인 |
|---|---|---|---|---|
| DistillDrive + 패킷 덤프 + 언어모델 학습·추론 | 3.8 | 1.13 (cu116) | transformers 4.46.3, peft 0.13.2, accelerate 1.0.1, bitsandbytes(8bit) | CPU에서 테스트 50개 통과 (torch 1.13.1, transformers 4.46.3, peft 0.13.2) |
| 라벨 생성 (Qwen2.5-VL) | 3.9 이상 (별도 환경) | 2.x | transformers 4.49 이상 | 3.8에서는 설치 불가. `--mock`만 3.8에서 동작 |

- Python 3.8에서 쓸 수 있는 마지막 버전: transformers 4.46.3, peft 0.13.2, accelerate 1.0.1 (torch는 2.4.1까지)
- 라벨 생성 결과는 `reasons.jsonl` 파일 하나라서, 그 파일만 3.8 환경으로 가져오면 학습부터는 3.8에서 진행한다.
- GPU에서 실제 확인하지 않은 것: bitsandbytes 8bit 로딩, 실제 Qwen1.5-1.8B-Chat 학습·생성 속도와 메모리.
