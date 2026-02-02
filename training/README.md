# PCB Defect Detection with YOLOv8 (Modularized)

이 프로젝트는 **YOLOv8**을 사용하여 PCB(Printed Circuit Board) 결함을 탐지하기 위한 모듈화된 파이프라인입니다. 동적 모듈 로딩, Stratified Split(층화 추출), 오프라인 증강(Offline Augmentation), 그리고 자동 CSV 제출 생성 기능을 제공합니다.

## 📂 프로젝트 구조 (Project Structure)

```text
.
├── config.yaml              # 통합 설정 파일 (하이퍼파라미터, 경로, 모듈 선택)
├── run_exp.py               # 메인 실행 스크립트 (학습 및 추론)
├── pretrained_weights/      # YOLO 사전 학습 가중치 저장소 (예: yolov8s.pt)
├── PCB_DATASET/             # 원본 데이터셋 (이미지 및 XML 라벨)
├── runs/                    # 실험 결과물 저장소 (로그, 가중치, CSV)
└── src/
    ├── datasets/
    │   ├── dataset.py       # 표준 데이터셋 모듈 (XML->YOLO 변환, 7:2:1 분할)
    │   └── dataset_aug.py   # 증강 데이터셋 모듈 (오프라인 Flip/Rotate 적용)
    ├── models/
    │   └── yolov8.py        # 모델 팩토리 (동적 로딩 및 가중치 관리)
    ├── train.py             # 트레이너 클래스 (커스텀 콜백, Manual TQDM 포함)
    ├── inference.py         # 추론 매니저 (CSV 생성 담당)
    └── utils.py             # 유틸리티 (시드 고정, 결과 폴더 정리)
```

## 🚀 빠른 시작 (Quick Start)

### 1. 필수 사항 (Prerequisites)
- Python 3.8 이상
- [uv](https://github.com/astral-sh/uv) (필수)

이 프로젝트는 `uv`를 통해 의존성을 관리합니다. `requirements.txt`는 별도로 존재하지 않으며, `pyproject.toml`과 `uv.lock`을 사용합니다.

```bash
# 의존성 설치 (uv.lock 기반 동기화)
uv sync
```

### 2. 가중치 준비 (Prepare Weights)
사용할 사전 학습 모델 가중치(예: `yolov8s.pt`)를 `pretrained_weights/` 폴더에 넣어주세요.
코드가 자동으로 해당 폴더에서 가중치를 찾습니다.

### 3. 실험 실행 (Run Experiment)
```bash
# 기본 설정으로 실행
python run_exp.py --config config.yaml

# uv를 통한 실행
uv run python run_exp.py --config config.yaml
```

이 명령어를 실행하면 다음 과정이 자동으로 진행됩니다:
1.  **데이터 준비**: XML을 YOLO 포맷으로 변환하고, Stratified Split(7:2:1)을 수행합니다.
2.  **학습 (Train)**: YOLOv8 모델을 학습 데이터로 파인튜닝(Fine-tune)합니다.
3.  **검증 (Validate)**: 검증 데이터셋으로 성능을 평가합니다.
4.  **추론 (Inference)**: 테스트 데이터셋에 대해 추론을 수행하고 `submission.csv`를 생성합니다.

## ⚙️ 설정 가이드 (`config.yaml`)

`config.yaml` 파일 하나로 파이프라인 전체를 제어할 수 있습니다. 각 항목에 대한 자세한 주석이 달려 있습니다.

```yaml
# 1. 실험 이름 (Experiment Name)
exp_name: "baseline"          # 결과 폴더: runs/baseline/

# 2. 모듈 선택 (Module Selection)
dataset_module: "dataset"     # 'dataset_aug'로 변경 시 오프라인 증강 적용
model_module: "yolov8"        # 다른 모델 구조로 확장 시 사용

# 3. 학습 파라미터 (Training Params)
epochs: 180
optimizer: 'AdamW'            # 옵션: 'SGD', 'Adam', 'AdamW', 'auto'
scheduler_type: 'cosine'      # 옵션: 'linear', 'cosine'
device: '0'                   # GPU Device ID (예: '0,1' 또는 'cpu')
seed: 42                      # 재현성을 위한 Random Seed

# 4. 증강 제어 (Augmentation Control) - YOLOv8 기본값 예시
mosaic: 1.0                   # Mosaic Prob (0.0 ~ 1.0)
scale: 0.5                    # Scale Gain
mixup: 0.0                    # Mixup Prob
# ... (기타 모든 YOLO 증강 인자 지원)
```

## 📊 결과물 (Outputs)

모든 결과물은 `runs/{exp_name}/` 폴더에 저장됩니다:

- **가중치 (Weights)**: `runs/{exp_name}/weights/best.pt` (최고 성능 모델)
- **추론 결과 (Inference)**: `runs/{exp_name}/inference/submission.csv` (제출용 파일)
- **탐지 시각화 (Detect)**: `runs/{exp_name}/detect/` (테스트 이미지 시각화 결과)
- **로그 (Logs)**: 학습 곡선 및 메트릭 기록 (WandB 지원).

## 🧩 확장성 (Extensibility)

- **새로운 데이터셋 전략**: `src/datasets/my_dataset.py`를 만들고 `get_dataset(config)` 함수를 구현하세요. 그리고 `config.yaml`에서 `dataset_module`을 변경하면 됩니다.
- **새로운 모델**: `src/models/my_model.py`를 만들고 `get_model(config)` 함수를 구현하세요. 그 후 `model_module`을 변경하세요.
