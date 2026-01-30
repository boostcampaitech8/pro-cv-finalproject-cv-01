# 구현 계획 - 모델 학습 (YOLO)

이 계획서는 PCB 결함 탐지 시스템의 YOLO 모델 학습 구현 단계를 설명합니다.

## 진행 상황

| 항목 | 상태 |
|------|------|
| 데이터셋 준비 | ⏳ 예정 |
| 데이터 전처리 | ⏳ 예정 |
| 모델 선정 | ⏳ 예정 |
| 학습 파이프라인 | ⏳ 예정 |
| 평가 및 검증 | ⏳ 예정 |
| 모델 내보내기 | ⏳ 예정 |

## 프로젝트 구조

```
training/
├── train.py            # 학습 스크립트
├── evaluate.py         # 평가 스크립트
├── export.py           # 모델 내보내기
├── data/
│   ├── images/         # 이미지 데이터
│   ├── labels/         # 라벨 데이터
│   └── dataset.yaml    # 데이터셋 설정
├── configs/
│   └── train_config.yaml
├── weights/            # 학습된 가중치
└── requirements.txt
```

## 데이터셋

| 항목 | 내용 |
|------|------|
| 데이터 형식 | YOLO 형식 (txt) |
| 클래스 | 결함 타입별 분류 |
| 분할 | train / val / test |

## 모델

| 항목 | 내용 |
|------|------|
| 베이스 모델 | YOLOv8 (또는 협의) |
| 입력 크기 | TBD |
| 출력 | 결함 위치 (bbox) + 타입 + 신뢰도 |

## 학습 설정

| 하이퍼파라미터 | 값 |
|---------------|-----|
| Epochs | TBD |
| Batch Size | TBD |
| Learning Rate | TBD |
| Optimizer | TBD |

## 평가 지표

- mAP (Mean Average Precision)
- Precision / Recall
- F1 Score
- Inference Time

## 다음 단계

1. 데이터셋 수집 및 라벨링
2. 데이터 전처리 파이프라인
3. 베이스 모델 선정
4. 학습 및 튜닝
5. 엣지 배포용 모델 내보내기
