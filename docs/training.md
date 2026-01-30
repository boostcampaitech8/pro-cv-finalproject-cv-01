# 구현 계획 - 모델 학습 (YOLO)

이 계획서는 PCB 결함 탐지 시스템의 YOLO 모델 학습 구현 단계를 설명합니다. 실제 구현은 `final_project/training` 디렉토리 내의 모듈형 파이프라인을 따릅니다.

## 진행 상황

| 항목 | 상태 | 비고 |
|------|------|-----|
| 데이터셋 준비 | ⏳ 완료 | `PCB_DATASET` (XML -> YOLO 자동 변환) |
| 데이터 전처리 | ⏳ 완료 | Stratified Split 적용 (심화 증강 기법 추가 탐색 예정) |
| 모델 선정 | ⏳ 완료 | YOLOv8 Baseline 적용 (다른 모델 아키텍처 비교 예정) |
| 학습 파이프라인 | ⏳ 완료 | `run_exp.py` 통합 스크립트 구현됨 |
| 평가 및 검증 | ⏳ 완료 | 모델 학습 후 자동 평가 및 시각화 수행 |
| 모델 내보내기 | ⏳ 대기 | 학습 완료 후 `best.pt` 생성 |

## 프로젝트 구조

실제 `training` 디렉토리의 구조를 반영했습니다.

```
training/
├── config.yaml              # 통합 설정 파일 (하이퍼파라미터 제어)
├── run_exp.py               # 메인 실행 스크립트
├── pretrained_weights/      # 사전 학습 가중치 (yolov8s.pt 등)
├── runs/                    # 실험 결과 (Logs, Weights, CSV)
├── PCB_DATASET/             # 데이터셋 폴더
└── src/
    ├── datasets/            # 데이터 로더 모듈
    ├── models/              # YOLO 모델 모듈
    └── train.py             # 트레이너 클래스
```

## 데이터셋

| 항목 | 내용 |
|------|------|
| 데이터 형식 | YOLO 포맷 (이미지 + .txt 라벨) |
| 클래스 | 6종 (missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper) |
| 분할 | Train(70%) / Val(20%) / Test(10%) (Stratified Split) |

## 모델

| 항목 | 내용 |
|------|------|
| 베이스 모델 | YOLOv8 (기본: yolov8s.pt) |
| 입력 크기 | 640x640 |
| 출력 | 결함 위치 (Bounding Box) + 클래스 + 신뢰도 |
| *Note* | *현재는 YOLOv8s를 사용 중이나, 추후 RT-DETR 등 다른 모델과의 성능 비교를 계획 중입니다.* |

## 학습 설정 (`config.yaml`)

| 하이퍼파라미터 | 값 |
|---------------|-----|
| Epochs | 200 |
| Batch Size | 16 |
| Learning Rate | Initial 0.001 (Cosine Scheduler) |
| Optimizer | AdamW |
| *Note* | *증강(Augmentation) 파라미터는 기본값을 사용 중이며, 성능 최적화를 위해 추가 실험 예정입니다.* |

## 평가 지표

- **mAP50 (Mean Average Precision @ IoU 0.5)**: 주요 성능 지표
- **mAP50-95**: 엄격한 기준의 성능 지표
- **Precision / Recall**: 정밀도 및 재현율
- **Inference Time**: 추론 속도

## 다음 단계

1. **환경 설정**: `uv sync` 로 의존성 설치
2. **가중치 준비**: `pretrained_weights` 폴더에 `yolov8s.pt` 다운로드
3. **학습 시작**: `uv run python run_exp.py --config config.yaml` 실행
4. **결과 확인**: `runs/baseline/` 폴더에서 `submission.csv` 및 시각화 결과 확인
5. **추가 실험**: 다양한 증강 기법(Mosaic, Mixup 등) 및 모델 아키텍처 비교 실험 수행 예정
