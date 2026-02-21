from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
import pendulum
from datetime import timedelta
import os

# Project settings
PROJECT_ROOT = "/workspace/final_project/training"
SCRIPTS_DIR = f"{PROJECT_ROOT}/scripts"
MODEL_CONFIG = f"{PROJECT_ROOT}/configs/config_qat.yaml"
PYTHON_BIN = "/workspace/final_project/training/.venv/bin/python"  # Project venv (has boto3/ultralytics/mlflow)

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
    description='PCB Defect Detection Retraining Pipeline (Sync -> Train/QAT -> Export)',
    schedule='0 18 * * 0',  # Every Sunday at 18:00 UTC (Mon 03:00 KST)
    start_date=pendulum.today('UTC').add(days=-1),
    tags=['podo', 'qat', 'mlops'],
    catchup=False
) as dag:


    # 1. Sync Data (Transient Mode)
    # Creates 'data_retrain.yaml' in dataset directory
    t1_sync = BashOperator(
        task_id='sync_data_from_s3',
        bash_command=f'{PYTHON_BIN} {SCRIPTS_DIR}/sync_data.py',
        cwd=PROJECT_ROOT
    )

    # 2. Train FP32 Model (Fine-tuning with new data)
    # Uses run_id for unique experiment name (robust across Airflow versions)
    run_id = "{{ run_id }}"
    
    t2_train_fp32 = BashOperator(
        task_id='train_fp32_model',
        bash_command=f'{PYTHON_BIN} {SCRIPTS_DIR}/run_exp.py --config {PROJECT_ROOT}/configs/config.yaml --data {PROJECT_ROOT}/PCB_DATASET/data_retrain.yaml --name {run_id}_fp32',
        cwd=PROJECT_ROOT
    )

    # 3. Train QAT Model
    # Inputs: data_retrain.yaml (from t1) and best.pt (from t2)
    fp32_best_pt = f"{PROJECT_ROOT}/runs/{run_id}_fp32/weights/best.pt"
    
    t3_train_qat = BashOperator(
        task_id='train_qat_model',
        bash_command=f'{PYTHON_BIN} {SCRIPTS_DIR}/train_qat.py --config {MODEL_CONFIG} --data {PROJECT_ROOT}/PCB_DATASET/data_retrain.yaml --name {run_id}_qat --weights {fp32_best_pt}',
        cwd=PROJECT_ROOT
    )

    # 4. Export to ONNX
    export_cmd = f"""
    RUN_DIR="{PROJECT_ROOT}/runs/qat/{run_id}_qat"
    BEST_PT="${{RUN_DIR}}/weights/best.pt"
    # Check for hybrid first (recalibrated)
    if [ -f "${{RUN_DIR}}/weights/best_hybrid.pt" ]; then
        BEST_PT="${{RUN_DIR}}/weights/best_hybrid.pt"
    elif [ -f "${{RUN_DIR}}/weights/best_qat_fallback.pt" ]; then
        BEST_PT="${{RUN_DIR}}/weights/best_qat_fallback.pt"
    fi
    
    ONNX_OUTPUT="${{RUN_DIR}}/weights/best.onnx"
    
    echo "Exporting ${{BEST_PT}}..."
    {PYTHON_BIN} {SCRIPTS_DIR}/export_qat.py --weights "${{BEST_PT}}" --base-weights {PROJECT_ROOT}/runs/yolov11m_640/weights/yolov11m_640.pt --output "${{ONNX_OUTPUT}}"
    """


    t3_export = BashOperator(
        task_id='export_onnx',
        bash_command=export_cmd,
        cwd=PROJECT_ROOT
    )
    
    # 5. Register to MLflow (Metadata Only)
    # Since we are not deploying to edge, we just register the artifact/model in MLflow for record.
    # We pass the ONNX path.
    register_cmd = f"""
    RUN_DIR="{PROJECT_ROOT}/runs/qat/{run_id}_qat"
    ONNX_PATH="${{RUN_DIR}}/weights/best.onnx"
    
    {PYTHON_BIN} {SCRIPTS_DIR}/register_model.py --model-path "${{ONNX_PATH}}" --tags "run_id={run_id},status=retrained"
    """
    
    t5_register = BashOperator(
        task_id='register_model',
        bash_command=register_cmd,
        cwd=PROJECT_ROOT
    )

    t1_sync >> t2_train_fp32 >> t3_train_qat >> t3_export >> t5_register
