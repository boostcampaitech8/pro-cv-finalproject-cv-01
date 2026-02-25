import argparse
import mlflow
from mlflow import MlflowClient
import os
import sys
import boto3
import json

# 프로젝트 루트 디렉토리를 경로에 추가합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="모델 아티팩트의 경로 (예: .onnx 파일)")
    parser.add_argument("--tags", default="", help="콤마(,)로 구분된 태그 쌍 (예: key1=value1,key2=value2)")
    parser.add_argument("--model-name", default="PCB_Defect_Detector", help="등록될 MLflow 모델 이름")
    args = parser.parse_args()

    # 1. MLflow 설정 (안정성을 위해 절대 경로 사용)
    mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
    mlflow.set_experiment("PCB_Retraining_Pipeline")
    client = MlflowClient()

    print(f"📦 모델 등록 작업 시작 (대상 파일: {args.model_path})...")

    # 등록 절차 수행을 위한 독립적인 Run 생성
    with mlflow.start_run(run_name="Model_Registration") as run:
        run_id = run.info.run_id

        # ONNX 모델을 아티팩트로 저장
        mlflow.log_artifact(args.model_path, artifact_path="model")

        # 명령어로 전달받은 태그(Meta attributes)들을 추가
        if args.tags:
            for tag in args.tags.split(","):
                if "=" in tag:
                    k, v = tag.split("=", 1)
                    mlflow.set_tag(k.strip(), v.strip())

    # 2. 방금 종료된 Run에 저장된 아티팩트 주소(URI)를 사용하여 모델 레지스트리에 모델을 공식적으로 등록합니다.
    # MLflow 3.x 버전의 변경을 대비하여 runs:/ 축약형 대신 완벽한 절대 경로 아티팩트 주소를 사용합니다.
    artifact_uri = client.get_run(run_id).info.artifact_uri
    model_source = f"{artifact_uri}/model/{os.path.basename(args.model_path)}"

    # 지정된 이름의 레지스트리가 없다면 생성합니다 (1회성).
    try:
        client.create_registered_model(args.model_name)
        print(f"📝 새로운 모델 모음(Registered Model)이 백엔드에 생성되었습니다: '{args.model_name}'")
    except mlflow.exceptions.MlflowException:
        pass  # 이미 해당 이름의 모델 분류가 존재하므로 넘어갑니다.

    print(f"📝 '{args.model_name}' 이름 아래로 새로운 버전(Version) 생성을 요청합니다...")
    version = client.create_model_version(
        name=args.model_name,
        source=model_source,
        run_id=run_id
    )

    print(f"✅ 모델 등록(버전 생성) 성공! 부여된 버전: {version.version}")
    print(f"   소스(출처) Run ID: {run_id}")
    print(f"   소스(출처) 물리 경로: {model_source}")
    
    # 등록 직후 Staging으로 전환 (평가 대기 상태)
    client.transition_model_version_stage(
        name=args.model_name,
        version=version.version,
        stage="Staging"
    )
    print(f"📋 모델 v{version.version} → Staging (평가 대기)")

    # 3. 엣지 디바이스(Edge Devices) 배포용 S3 버킷에 직접 바이너리 파일 업로드
    s3_bucket = os.environ.get("S3_BUCKET_NAME", "pcb-data-storage")
    if not s3_bucket:
        print("⚠️ 환경 변수 S3_BUCKET_NAME 이 설정되지 않았습니다. 외부 업로드 단계를 건너뜁니다.")
        return

    try:
        s3_client = boto3.client('s3')
        
        # 실제 ONNX 파일을 S3 레지스트리용 폴더에 업로드
        onnx_filename = os.path.basename(args.model_path)
        s3_key = f"models/candidates/{version.version}_{onnx_filename}"
        print(f"☁️ 클라우드(s3://{s3_bucket}/{s3_key})로 {args.model_path} 업로드 진행 중...")
        s3_client.upload_file(args.model_path, s3_bucket, s3_key)
        
        # 엣지 디바이스가 바라볼 엔드포인트 파일(latest.json) 목록 갱신 및 업로드
        import time
        # --tags 인자에서 파싱한 태그를 dict로 변환하여 포함
        # Jetson의 조건 필터(REQUIRED_TAG_KEY/VALUE)가 이 필드를 읽습니다.
        tags_dict = {}
        if args.tags:
            for tag in args.tags.split(","):
                if "=" in tag:
                    k, v = tag.split("=", 1)
                    tags_dict[k.strip()] = v.strip()

        latest_info = {
            "version": version.version,
            "s3_key": s3_key,
            "run_id": run_id,
            "tags": tags_dict,
            "timestamp": int(time.time() * 1000)
        }
        latest_json_path = "/tmp/latest.json"
        with open(latest_json_path, 'w') as f:
            json.dump(latest_info, f, indent=4)
            
        latest_s3_key = "models/candidates/latest.json"
        print(f"☁️ 클라우드 동기화 상태 파일 갱신 시도 중 (s3://{s3_bucket}/{latest_s3_key})...")
        s3_client.upload_file(latest_json_path, s3_bucket, latest_s3_key)
        
        print("✅ 모든 아티팩트의 S3 클라우드 업로드 파이프라인이 완료되었습니다!")

    except Exception as e:
        print(f"❌ 외부 S3 저장소 업로딩 중 예상치 못한 에러 발생: {e}")

if __name__ == "__main__":
    main()
