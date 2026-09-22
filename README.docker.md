# DistillDrive Docker 환경

이 구성은 원본 설치 문서의 CUDA 11.6/PyTorch 1.13 환경 대신 ARM64 NVIDIA GB10에서 실행 가능한 `nvcr.io/nvidia/pytorch:26.07-py3` 컨테이너를 사용한다. 원본 소스는 커밋 `461f0722ffac5df7c5a2247bfda67ecdcb8b7952`에 고정되어 있다.

## 1. 이미지 빌드와 확인

```bash
cd /workspace/lab-infra/distilldrive
docker compose build
docker compose run --rm workspace python docker/smoke_test.py
```

두 번째 명령은 GPU, PyTorch, MMCV, MMDetection, flash-attn 및 DistillDrive CUDA 확장을 확인한다.

## 2. 데이터 준비

nuScenes 데이터와 CAN bus expansion을 호스트의 `data/nuscenes` 아래에 둔다. 컨테이너에서는 `/workspace/DistillDrive/data/nuscenes`로 보인다. 원본 문서가 기대하는 기본 구조는 다음과 같다.

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
└── can_bus/
```

메타데이터와 K-means anchor는 다음 명령으로 생성한다.

```bash
docker compose run --rm workspace bash scripts/create_data.sh
docker compose run --rm workspace bash scripts/kmeans.sh
```

데이터셋은 용량과 라이선스 때문에 자동 다운로드하지 않는다.

## 3. 체크포인트와 실행

사전 학습 가중치는 호스트의 `checkpoint`에 저장한다. ResNet-50 backbone 예시는 다음과 같다.

```bash
wget https://download.pytorch.org/models/resnet50-19c8e357.pth \
  -O checkpoint/resnet50-19c8e357.pth
```

대화형 셸:

```bash
docker compose run --rm workspace
```

단일 GPU 평가 예시:

```bash
docker compose run --rm workspace \
  bash scripts/test.sh \
  projects/configs/stage2/distilldrive_stage2_label.py \
  checkpoint/distilldrive_stage2_label.pth \
  1
```

학습 예시:

```bash
docker compose run --rm workspace \
  bash scripts/train.sh projects/configs/stage2/distilldrive_stage2_label.py 1
```

`data`, `checkpoint`, `work_dirs`, `vis`는 호스트에 유지되므로 컨테이너를 삭제해도 남는다. 기본 이미지나 최종 이미지 이름은 각각 `DISTILLDRIVE_BASE_IMAGE`, `DISTILLDRIVE_IMAGE` 환경 변수로 바꿀 수 있다.

## 알려진 차이

- Python 3.12/Blackwell을 지원하지 않는 업스트림 고정 버전은 동일 API 세대의 설치 가능한 버전으로 조정했다.
- DistillDrive 고유 CUDA 연산은 GB10의 `sm_121` 타깃으로 소스 컴파일한다.
- `mmcv-full` 1.7.1은 최신 PyTorch C++ API에서 컴파일되지 않는다. DistillDrive는 `mmcv.ops`를 직접 사용하지 않으므로 MMDetection의 registry import만 통과시키는 shim을 포함한다. 다른 코드 경로에서 MMCV native op를 호출하면 명확한 `RuntimeError`로 중단된다.
- nuScenes 전체 데이터와 모델 가중치가 없으면 실제 평가·학습까지는 실행할 수 없다.
