#!/bin/bash

# YOLOv11 Knowledge Distillation Experiment (Full: Feature + Logit)
# Teacher: yolov11m (960px)
# Student: yolov11n (Pretrained)
# Config: Aligned with Baseline (AdamW, lr=0.001, 200e)
# Feature KD: CWD (Channel-wise Distillation)

python scripts/train_kd_feature.py \
  --name kd_beta \
  --teacher runs/v11x_1280_fix/weights/best.pt \
  --student runs/yolov11n_960/weights/best.pt \
  --data PCB_DATASET/data.yaml \
  --epochs 100 \
  --patience 20 \
  --imgsz 960 \
  --optimizer AdamW \
  --lr0 0.001 \
  --cos-lr \
  --alpha-box 0.0000 \
  --alpha-cls 0.000 \
  --beta 1.0 \
  --feature-loss-type cwd \
  --batch 8 \