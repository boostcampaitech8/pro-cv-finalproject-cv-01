# QAT (Quantization-Aware Training) 경량화

## 개요

- **목표**: INT8 양자화 시 mAP 손실 최소화
- **방법**: NVIDIA pytorch-quantization 사용
- **대상**: YOLOv8s 모델
- **TensorRT 호환**: Q/DQ 노드 포함 ONNX export로 TensorRT INT8 최적 변환 지원

## 현재 벤치마크 결과 (Jetson + TensorRT)

| 정밀도 | 양자화 | FPS | mAP50 |
|--------|--------|------|-------|
| FP16 | 없음 | 43.13 | 0.9265 |
| INT8 | PTQ | 58.81 | 0.9122 |

## QAT 목표

| 지표 | 현재 (PTQ) | 목표 (QAT) | 기준 (FP16) |
|------|-----------|-----------|-------------|
| mAP50 | 0.9122 | **0.92+** | 0.9265 |
| FPS | 58.81 | **58+** | 43.13 |
| mAP 하락 | 1.5% | **< 0.5%** | 0% |

## 구현 파이프라인

```
1. 일반 학습 (FP32)
   └── best.pt 생성

2. QAT Fine-tuning
   ├── Pre-trained 모델 로드
   ├── Conv2d → QuantConv2d 교체
   ├── Calibration (Histogram 기반)
   └── Fine-tuning (30 epochs)

3. Export
   └── ONNX (Q/DQ 노드 포함)

4. Jetson 배포
   └── TensorRT INT8 변환
```

## 파일 구조

```
training/
├── config_qat.yaml              # QAT 전용 설정
├── run_qat.py                   # QAT 실행 스크립트
├── verify_qat.py                # QAT 변환 검증 스크립트
└── src/
    ├── train_qat.py             # QATTrainer 클래스
    ├── models/
    │   └── yolov8s_qat.py       # QAT 모델 래퍼
    └── quantization/            # QAT 유틸리티
        ├── __init__.py
        ├── qat_utils.py         # Q/DQ 노드 삽입, calibration
        └── export_qat.py        # ONNX export
```

## Quick Start (전체 흐름)

> **주의**: QAT는 **NVIDIA GPU가 있는 서버에서만** 실행 가능합니다. Mac에서는 pytorch-quantization이 설치되지 않습니다.

```bash
# 1. 서버 접속
ssh pcb-defect
cd ~/pro-cv-finalproject-cv-01/training

# 2. 의존성 설치 (QAT 포함)
uv sync --extra qat

# 3. (선택) 양자화 레이어 적용 검증
uv run python verify_qat.py

# 4. QAT 학습 실행
uv run python run_qat.py --config config_qat.yaml

# 5. 결과물
# - runs/qat/qat_yolov8s/weights/best.pt (QAT 모델)
# - runs/qat/qat_yolov8s/weights/best_qat.onnx (TensorRT용)

# 6. Jetson 배포
trtexec --onnx=best_qat.onnx --saveEngine=yolov8s_int8.engine --int8
```

---

## 사전 준비

### 1. 의존성 설치

**IMPORTANT**: `pytorch-quantization`은 NVIDIA NGC 저장소에서만 제공되며, **NVIDIA GPU가 있는 환경에서만** 설치 가능합니다.

**방법 A: optional-dependencies 사용 (권장)**
```bash
cd training
uv sync --extra qat
```

**방법 B: 수동 설치**
```bash
cd training
uv pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
uv sync
```

`pyproject.toml` 설정:
```toml
[project.optional-dependencies]
qat = ["pytorch-quantization>=2.1.3"]

[[tool.uv.index]]
name = "nvidia"
url = "https://pypi.ngc.nvidia.com"
explicit = true
```

**설치 확인**:
```bash
uv run python -c "from pytorch_quantization import quant_modules; print('✅ pytorch-quantization 설치 성공')"
```

### 2. 양자화 레이어 적용 검증

QAT 학습 전에 `QuantConv2d`가 제대로 적용되는지 확인:

```bash
uv run python verify_qat.py
```

**성공 시 출력**:
```
✅ 성공: QuantConv2d 레이어가 발견되었습니다.
   QAT가 정상적으로 적용될 준비가 되었습니다.
```

**실패 시**: `initialize_quantization()` 호출 순서 확인 필요

### 3. 일반 학습 수행 (best.pt 필요)

```bash
cd training
uv run python run_exp.py --config config.yaml
```

결과: `runs/baseline_v8s/weights/best.pt`

## 실행 방법

### QAT Fine-tuning

```bash
cd training
uv run python run_qat.py --config config_qat.yaml
```

### 주요 설정 (config_qat.yaml)

```yaml
qat:
  # Pre-trained 모델 경로
  pretrained_path: "./runs/baseline_v8s/weights/best.pt"

  # Calibration 설정
  calibration:
    num_batches: 100      # Calibration에 사용할 배치 수
    method: "histogram"   # histogram 또는 max

  # 양자화 설정
  quantization:
    num_bits: 8           # 양자화 비트 (INT8)
    quant_conv: true      # Conv2d 양자화
    quant_linear: true    # Linear 양자화
    weight_per_channel: true  # 채널별 weight 양자화

  # Fine-tuning 설정
  finetune:
    epochs: 30            # QAT fine-tuning 에폭
    lr0: 0.0001           # 낮은 학습률 사용
    lrf: 0.1              # 최종 LR 비율
```

## 결과 확인

### 로컬 검증 (PyTorch)

```bash
cd training
uv run python -c "
from ultralytics import YOLO
model = YOLO('runs/qat/qat_yolov8s/weights/best.pt')
metrics = model.val(data='PCB_DATASET/data.yaml')
print(f'QAT mAP50: {metrics.box.map50:.4f}')
"
```

### ONNX Export 결과

QAT 학습 완료 후 자동으로 ONNX가 export됨:
- `runs/qat/qat_yolov8s/weights/best_qat.onnx`

### Jetson 배포 (TensorRT INT8)

```bash
# Jetson에서 실행
trtexec --onnx=yolov8s_qat.onnx --saveEngine=yolov8s_qat_int8.engine --int8
```

## 주의 사항

1. **메모리**: QAT는 메모리 사용량이 증가함. `batch_size` 감소 필요 (16 → 8)

2. **Detect Head**: 마지막 Detect 레이어는 양자화 제외 권장 (정확도에 민감)

3. **BatchNorm**: Conv-BN fusion 적용됨

4. **호환성**: pytorch-quantization 버전과 PyTorch/CUDA 버전 호환 확인 필요
   - PyTorch >= 2.0 권장
   - CUDA 11.8 or 12.x 지원

5. **Calibration 데이터**: 학습 데이터의 대표적인 샘플 사용 (100 batches 권장)

## 문제 해결

### Q: pytorch-quantization 설치 오류

```bash
# CUDA 버전에 맞는 pytorch-quantization 설치
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

### Q: Calibration 중 메모리 부족

`config_qat.yaml`에서 `calibration.num_batches`를 줄임 (100 → 50)

### Q: TensorRT 변환 실패

ONNX 파일에 Q/DQ 노드가 포함되어 있는지 확인:
```bash
python -c "import onnx; m=onnx.load('model.onnx'); print([n.op_type for n in m.graph.node if 'Quant' in n.op_type])"
```

## 참고 자료

- [NVIDIA pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)
- [TensorRT QAT Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#qat)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
