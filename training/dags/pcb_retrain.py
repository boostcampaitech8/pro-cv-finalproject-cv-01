from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
import pendulum
from datetime import timedelta
import os

# 프로젝트 설정 관리
PROJECT_ROOT = "/workspace/pro-cv-finalproject-cv-01/training"
SCRIPTS_DIR = f"{PROJECT_ROOT}/scripts"
MODEL_CONFIG = f"{PROJECT_ROOT}/configs/config_qat.yaml"
PYTHON_BIN = "/workspace/pro-cv-finalproject-cv-01/training/.venv/bin/python"  # 프로젝트 가상환경 (boto3/ultralytics/mlflow 포함)

# AWS 자격증명 — Airflow subprocess에서도 접근 가능하도록 환경변수 설정
AWS_ENV = {
    'HOME': '/root',
    'AWS_SHARED_CREDENTIALS_FILE': '/root/.aws/credentials',
    'AWS_CONFIG_FILE': '/root/.aws/config',
}

default_args = {
    'owner': 'podo_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pcb_retrain_pipeline',
    default_args=default_args,
    description='PCB 결함 탐지 재학습 파이프라인 (S3 동기화 -> FP32 학습 -> QAT 학습 -> ONNX 추출)',
    schedule='0 18 * * 0',  # 매주 일요일 UTC 18:00 (한국 시간 월요일 03:00)
    start_date=pendulum.today('UTC').add(days=-1),
    tags=['podo', 'qat', 'mlops'],
    catchup=False
) as dag:


    # 1. 데이터 동기화 (임시 저장소 사용)
    # dataset 디렉토리에 'data_retrain.yaml'을 생성합니다.
    t1_sync = BashOperator(
        task_id='sync_data_from_s3',
        bash_command=f'{PYTHON_BIN} {SCRIPTS_DIR}/sync_data.py',
        cwd=PROJECT_ROOT,
        env=AWS_ENV
    )

    # 2. FP32 모델 학습 (새로운 데이터 셋 기반 재학습)
    # 파일 경로 충돌을 방지하기 위해 특수문자가 없는 Airflow의 'ts_nodash'를 실험명으로 사용합니다.
    run_id = "{{ ts_nodash }}"
    
    t2_train_fp32 = BashOperator(
        task_id='train_fp32_model',
        bash_command=f'{PYTHON_BIN} {SCRIPTS_DIR}/run_exp.py --config {PROJECT_ROOT}/configs/config.yaml --data {PROJECT_ROOT}/PCB_DATASET/data_retrain.yaml --name {run_id}_fp32',
        cwd=PROJECT_ROOT
    )

    # 3. 양자화 인지 학습 (QAT)
    # 입력값: data_retrain.yaml (t1에서 생성) 및 best.pt (t2에서 생성)
    fp32_best_pt = f"{PROJECT_ROOT}/runs/{run_id}_fp32/weights/best.pt"
    
    t3_train_qat = BashOperator(
        task_id='train_qat_model',
        bash_command=f'{PYTHON_BIN} {SCRIPTS_DIR}/train_qat.py --config {MODEL_CONFIG} --data {PROJECT_ROOT}/PCB_DATASET/data_retrain.yaml --name {run_id}_qat --weights {fp32_best_pt}',
        cwd=PROJECT_ROOT
    )

    # 4. ONNX 모델 병합 및 추출
    export_cmd = f"""
    RUN_DIR="{PROJECT_ROOT}/runs/qat/{run_id}_qat"
    BEST_PT="${{RUN_DIR}}/weights/best.pt"
    # 재보정(Calibration)을 거친 최적의 하이브리드 모델을 우선 선택합니다.
    if [ -f "${{RUN_DIR}}/weights/best_hybrid.pt" ]; then
        BEST_PT="${{RUN_DIR}}/weights/best_hybrid.pt"
    elif [ -f "${{RUN_DIR}}/weights/best_qat_fallback.pt" ]; then
        BEST_PT="${{RUN_DIR}}/weights/best_qat_fallback.pt"
    fi
    
    ONNX_OUTPUT="${{RUN_DIR}}/weights/best.onnx"
    
    echo "ONNX 변환 진행 중: ${{BEST_PT}}..."
    {PYTHON_BIN} {SCRIPTS_DIR}/export_qat.py --weights "${{BEST_PT}}" --base-weights {fp32_best_pt} --output "${{ONNX_OUTPUT}}"
    """


    t3_export = BashOperator(
        task_id='export_onnx',
        bash_command=export_cmd,
        cwd=PROJECT_ROOT
    )
    
    # 5. MLflow 모델 레지스트리 등록 (메타데이터만 관리)
    # 엣지 런타임에 직접 배포하지 않고, MLflow 아티팩트로만 기록을 남깁니다.
    # 등록 대상으로 추출된 ONNX 파일 경로를 전달합니다.
    register_cmd = f"""
    RUN_DIR="{PROJECT_ROOT}/runs/qat/{run_id}_qat"
    ONNX_PATH="${{RUN_DIR}}/weights/best.onnx"
    
    # 모델 설정을 통해 사용된 YOLO 버전을 파악합니다.
    # config.yaml의 model_module 값을 파싱하여 yolo_version 태그로 부여
    YOLO_VER=$(grep -oP '^model_module:\\s*"\\K[^"]+' {PROJECT_ROOT}/configs/config.yaml || echo "unknown")
    
    {PYTHON_BIN} {SCRIPTS_DIR}/register_model.py --model-path "${{ONNX_PATH}}" --tags "run_id={run_id},status=retrained,yolo_version=${{YOLO_VER}}"
    """
    
    t5_register = BashOperator(
        task_id='register_model',
        bash_command=register_cmd,
        cwd=PROJECT_ROOT,
        env=AWS_ENV
    )

    t1_sync >> t2_train_fp32 >> t3_train_qat >> t3_export >> t5_register

