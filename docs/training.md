# 모델 학습

PCB 결함 탐지 YOLO 모델 학습 모듈입니다.

## 개요

**역할:** YOLOv8 모델을 PCB 결함 데이터셋으로 학습

## 기술 스택

| 구분 | 기술 |
|------|------|
| Python | 3.8 이상 |
| 모델 | YOLOv8s |
| 학습 기법 | QAT (Quantization-Aware Training) |
| 패키지 관리 | uv |

## 프로젝트 구조

```
training/
├── config.yaml              # 통합 설정 파일
├── run_exp.py               # 메인 실행 스크립트
├── pretrained_weights/      # 사전 학습 가중치
├── PCB_DATASET/             # 원본 데이터셋
├── runs/                    # 실험 결과물
└── src/
    ├── datasets/
    │   ├── dataset.py       # 표준 데이터셋
    │   └── dataset_aug.py   # 증강 데이터셋
    ├── models/
    │   └── yolov8.py        # 모델 팩토리
    ├── train.py             # 트레이너 클래스
    ├── inference.py         # 추론 매니저
    └── utils.py             # 유틸리티
```

## 데이터셋

**PCB_DATASET:**
- 클래스: 6종 (scratch, hole, contamination, crack 등)
- 포맷: XML 라벨 → YOLO 포맷 자동 변환
- 분할: Stratified Split (7:2:1)

## 모델

**YOLOv8s:**
- 사전 학습 가중치: `yolov8s.pt`
- QAT 적용: Jetson 추론 최적화
- 학습된 모델: `qat_yololv8s_612_best.pt`

## 학습 설정 (config.yaml)

```yaml
# 실험 이름
exp_name: "baseline"

# 모듈 선택
dataset_module: "dataset"
model_module: "yolov8"

# 학습 파라미터
epochs: 180
optimizer: 'AdamW'
scheduler_type: 'cosine'
device: '0'
seed: 42

# 증강 제어
mosaic: 1.0
scale: 0.5
mixup: 0.0
```

## 평가 지표

| 지표 | 설명 |
|------|------|
| mAP50 | IoU 0.5에서의 평균 정밀도 |
| mAP50-95 | IoU 0.5~0.95 평균 정밀도 |

## 실행 방법

### 의존성 설치

```bash
cd training
uv sync
```

### 가중치 준비

사전 학습 모델 가중치를 `pretrained_weights/` 폴더에 넣습니다.

### 학습 실행

```bash
# 기본 설정
python run_exp.py --config config.yaml

# uv를 통한 실행
uv run python run_exp.py --config config.yaml
```

**자동 진행 과정:**
1. 데이터 준비 (XML → YOLO 변환, 7:2:1 분할)
2. 학습 (Train)
3. 검증 (Validate)
4. 추론 (Inference)

## 결과물

모든 결과물은 `runs/{exp_name}/` 폴더에 저장됩니다:

| 항목 | 경로 | 설명 |
|------|------|------|
| 가중치 | `runs/{exp_name}/weights/best.pt` | 최고 성능 모델 |
| 추론 결과 | `runs/{exp_name}/inference/submission.csv` | 제출용 파일 |
| 시각화 | `runs/{exp_name}/detect/` | 테스트 이미지 결과 |
| 로그 | WandB | 학습 곡선, 메트릭 기록 |

## 확장성

### 새로운 데이터셋 전략

`src/datasets/my_dataset.py`를 만들고 `get_dataset(config)` 함수를 구현한 후, `config.yaml`에서 `dataset_module`을 변경합니다.

### 새로운 모델

`src/models/my_model.py`를 만들고 `get_model(config)` 함수를 구현한 후, `config.yaml`에서 `model_module`을 변경합니다.

## 상세 문서

전체 상세 구현은 `training/README.md` 참조.
