import os
import time
import json
import boto3
import sys
import subprocess
from datetime import datetime

# Import local modules (assuming in serving/edge)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from dotenv import load_dotenv
load_dotenv()

# MLflow for Edge (simplified or standard client)
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow package not found. Continuing without MLflow tracking.")


class ModelUpdater:
    def __init__(self, interval=None):
        self.current_version = "0"
        os.makedirs(config.MODELS_DIR, exist_ok=True)

        # S3 Client setup
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ap-southeast-2")
        )
        self.bucket = "pcb-data-storage"
        self.latest_json_key = "models/candidates/latest.json"

        # Configure MLflow
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            print(f"📡 MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
            self.client = MlflowClient()
        else:
            self.client = None

        self.model_name = os.environ.get("MLFLOW_MODEL_NAME", "PCB_Defect_Detector")
        self.edge_device_id = os.environ.get("EDGE_DEVICE_ID", "edge-jetson-01")
        # interval은 config 기본값 사용, 명시적으로 넘기면 덮어씀
        self.interval = interval if interval is not None else config.MODEL_POLL_INTERVAL

    # ── 유틸리티 ────────────────────────────────────────────────────────────

    def _wait_until_idle(self, context: str = ""):
        """`config.INFERENCE_BUSY_FLAG_PATH`가 사라질 때까지 대기합니다.

        context는 로그 메시지에 포함되어 현재 무슨 작업을 대기하고 있는지
        표시합니다. 예: "download start", "build start" 등.
        """
        while os.path.exists(config.INFERENCE_BUSY_FLAG_PATH):
            desc = f" ({context})" if context else ""
            print(f"🔒 추론 바쁨{desc} — 대기 중...")
            self._kill_trtexec_if_running()
            time.sleep(10)

    def _kill_trtexec_if_running(self):
        """현재 실행중인 trtexec 프로세스를 찾아 강제 종료하고 관련 빌드 파일 삭제."""
        killed = False
        for pid in os.popen("pgrep -f trtexec").read().split():
            try:
                os.kill(int(pid), 9)
                print(f"⛔ trtexec PID {pid} 강제 종료 (업데이트 일시 중단)")
                killed = True
            except Exception:
                pass
        if not killed:
            # nothing to do
            return
        # if we killed something, also remove BUILDING_FLAG_PATH if exists
        if os.path.exists(config.BUILDING_FLAG_PATH):
            try:
                os.remove(config.BUILDING_FLAG_PATH)
                print("  빌드 플래그도 함께 삭제됨")
            except Exception:
                pass
        # 제거할 빌드 파일 목록이 있으면 삭제
        if hasattr(self, 'current_build_files') and self.current_build_files:
            for fpath in self.current_build_files:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                        print(f"  빌드 중이던 파일 삭제: {fpath}")
                    except Exception as e:
                        print(f"  파일 삭제 실패: {fpath} ({e})")
            self.current_build_files = []

    # ── 조건 필터 ────────────────────────────────────────────────────────────

    def _meets_condition(self, meta: dict) -> bool:
        """
        latest.json의 tags 필드에서 배포 조건을 확인합니다.
        config.REQUIRED_TAG_KEY가 비어 있으면 모든 모델을 허용합니다.
        """
        if not config.REQUIRED_TAG_KEY:
            return True
        tags = meta.get("tags", {})
        actual = tags.get(config.REQUIRED_TAG_KEY)
        if actual != config.REQUIRED_TAG_VALUE:
            print(
                f"⏭  조건 불충족 → 스킵: "
                f"{config.REQUIRED_TAG_KEY}={actual!r} "
                f"(필요: {config.REQUIRED_TAG_VALUE!r})"
            )
            return False
        return True

    # ── 상태/중단 헬퍼 ──────────────────────────────────────────────────────────

    def print_status(self):
        """현재 빌드 플래그 및 프로세스 상태를 출력"""
        print(f"BUILDING_FLAG_PATH: {config.BUILDING_FLAG_PATH}")
        if os.path.exists(config.BUILDING_FLAG_PATH):
            try:
                with open(config.BUILDING_FLAG_PATH, 'r') as f:
                    print(f"  내용: {f.read().strip()}")
            except Exception:
                pass
        else:
            print("  플래그 없음")
        # 프로세스 검색
        procs = []
        for line in os.popen("pgrep trtexec -a").read().splitlines():
            procs.append(line)
        if procs:
            print("현재 실행 중인 trtexec 프로세스:")
            for p in procs:
                print("  ", p)
        else:
            print("실행 중인 trtexec 없음")

    def abort_build(self):
        """진행 중인 TRT 빌드를 중단하고 관련 파일을 정리합니다."""
        if not os.path.exists(config.BUILDING_FLAG_PATH):
            print("빌드 플래그가 없음. 중단할 작업이 없습니다.")
            return
        try:
            with open(config.BUILDING_FLAG_PATH, 'r') as f:
                info = f.read().strip()
        except Exception:
            info = "(읽기 실패)"
        print(f"빌드 플래그 발견: {info}")
        # trtexec 프로세스 종료
        killed = False
        for pid in os.popen("pgrep trtexec").read().split():
            try:
                os.kill(int(pid), 9)
                print(f"  trtexec PID {pid} 종료")
                killed = True
            except Exception:
                pass
        if not killed:
            print("  실행 중인 trtexec 프로세스를 찾지 못함")
        # 관련 파일 삭제 (version 번호 포함된 onnx/engine)
        for fname in os.listdir(config.MODELS_DIR):
            if fname.startswith("v") and (fname.endswith(".onnx") or fname.endswith(".engine")):
                path = os.path.join(config.MODELS_DIR, fname)
                try:
                    os.remove(path)
                    print(f"  삭제: {path}")
                except Exception as e:
                    print(f"  삭제 실패: {path} ({e})")
        # 플래그 삭제
        try:
            os.remove(config.BUILDING_FLAG_PATH)
            print("빌드 플래그 삭제")
        except Exception as e:
            print(f"빌드 플래그 삭제 실패: {e}")
        print("빌드 중단 및 정리 완료")

    # ── S3 폴링 ──────────────────────────────────────────────────────────────

    def check_for_updates(self):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] S3 업데이트 확인 중...")
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.latest_json_key)
            latest_meta = json.loads(response['Body'].read().decode('utf-8'))
            latest_v = str(latest_meta.get('version', '0'))

            if latest_v == str(self.current_version):
                print(f"  최신 버전 유지 중 (v{self.current_version})")
                return

            print(f"🌟 새 버전 발견: v{latest_v} (현재: v{self.current_version})")

            if not self._meets_condition(latest_meta):
                return  # 조건 불충족 → 스킵

            self.download_and_build(latest_v, latest_meta)

        except Exception as e:
            if "NoSuchKey" not in str(e):
                print(f"⚠️  업데이트 확인 중 오류: {e}")

    # ── 다운로드 & TRT 빌드 ──────────────────────────────────────────────────

    def download_and_build(self, version, meta):
        s3_key = meta['s3_key']
        local_onnx_path   = os.path.join(config.MODELS_DIR, f"v{version}.onnx")
        local_engine_path = os.path.join(config.MODELS_DIR, f"v{version}.engine")

        # ① 이미 같은 버전의 엔진이 있으면 빌드 스킵하고 바로 평가
        if os.path.exists(local_engine_path):
            print(f"⚡ v{version}.engine 이미 존재 → 빌드 스킵, 바로 평가")
            self._evaluate_and_promote(local_engine_path, version, meta)
            return

        # ② 중복 빌드 방지: BUILDING_FLAG가 있으면 다른 updater 인스턴스나
        #     이전 작업이 아직 trtexec를 돌리고 있다는 뜻입니다.
        #     이 경우 지금 수행 중인 주기를 건너뜁니다.
        if os.path.exists(config.BUILDING_FLAG_PATH):
            print("⚠️  이미 TRT 빌드 진행 중 (BUILDING_FLAG 존재) → 이번 폴링 주기 스킵")
            return

        # ③ ONNX 다운로드
        #    다운로드 중에는 바쁨 플래그가 발생해도 중단하기 어렵기 때문에
        #    시작 전에 한 번 확인만 합니다.
        self.current_build_files = [local_onnx_path, local_engine_path]
        self._wait_until_idle("download start")
        print(f"📥 S3에서 다운로드 중: {s3_key} → {local_onnx_path}")
        try:
            self.s3_client.download_file(self.bucket, s3_key, local_onnx_path)
            print(f"✅ 다운로드 완료: {local_onnx_path}")
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            self.report_status(version, "download_failed", str(e), run_id=meta.get("run_id"))
            return

        # ④ TRT 엔진 빌드
        #    BUILDING_FLAG: updater 내부 전용 (중복 빌드 방지).
        #    inference_worker는 이 플래그를 읽지 않으며,
        #    빌드 중에도 기존 엔진으로 추론을 계속 수행합니다.
        #    단, 사용자가 선택한 정책에 따라 "추론이 바쁠 때 빌드를 대기"하도록 구성되어
        #    있으면, 실제 trtexec 실행 전에 `INFERENCE_BUSY_FLAG_PATH`가 없어질 때까지 대기합니다.
        #    (현재 정책: wait-until-idle before build)
        if os.path.exists(config.INFERENCE_BUSY_FLAG_PATH):
            print("🔒 추론이 바쁨 — 빌드 대기 중...")
        while os.path.exists(config.INFERENCE_BUSY_FLAG_PATH):
            print("🔒 추론이 계속 바쁨 — 10초 후 재확인")
            time.sleep(10)
        # ④ TRT 엔진 빌드
        print(f"🔨 TensorRT 엔진 빌드 준비 (workspace={config.TRT_WORKSPACE_MB}MB, "
              f"avgTiming={config.TRT_AVG_TIMING_ITERS})...")
        with open(config.BUILDING_FLAG_PATH, 'w') as f:
            f.write(f"building v{version}")

        # 빌드는 장시간 걸리므로 바쁨 플래그가 켜지면 즉시 프로세스를 종료
        self.current_build_files = [local_onnx_path, local_engine_path]
        self._wait_until_idle("build start")
        try:
            cmd = (
                f"trtexec"
                f" --onnx={local_onnx_path}"
                f" --saveEngine={local_engine_path}"
                f" --int8"
                f" --fp16"
                f" --workspace={config.TRT_WORKSPACE_MB}"
                f" --minTiming={config.TRT_MIN_TIMING_ITERS}"
                f" --avgTiming={config.TRT_AVG_TIMING_ITERS}"
            )
            proc = subprocess.Popen(cmd, shell=True)
            # 모니터링 루프
            while True:
                time.sleep(5)
                if proc.poll() is not None:
                    break
                if os.path.exists(config.INFERENCE_BUSY_FLAG_PATH):
                    print("🔒 추론 중 감지됨 — 빌드 프로세스 종료")
                    proc.kill()
                    proc.wait()
                    print("🔨 빌드 중단됨 (추론 바쁨)")
                    return
            if proc.returncode == 0:
                print("✅ TRT 엔진 빌드 성공!")
            else:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        except subprocess.TimeoutExpired:
            print(f"❌ TRT 빌드 타임아웃 ({config.TRT_BUILD_TIMEOUT_S}초 초과)")
            self.report_status(version, "build_timeout", "trtexec timeout", run_id=meta.get("run_id"))
            return
        except subprocess.CalledProcessError as e:
            print(f"❌ TRT 빌드 실패: {e}")
            self.report_status(version, "build_failed", str(e), run_id=meta.get("run_id"))
            return
        finally:
            # 성공/실패 무관하게 FLAG 제거
            if os.path.exists(config.BUILDING_FLAG_PATH):
                os.remove(config.BUILDING_FLAG_PATH)

        # ⑤ 평가 & 교체
        self._evaluate_and_promote(local_engine_path, version, meta)

    # ── 평가 후 교체 ─────────────────────────────────────────────────────────

    def _evaluate_and_promote(self, engine_path, version, meta):
        new_map     = self.evaluate_engine(engine_path)
        current_map = self.evaluate_engine(config.MODEL_PATH) if os.path.exists(config.MODEL_PATH) else 0.0

        print(f"⚖️  Shadow 평가 결과:")
        print(f"   현재 모델 mAP50 : {current_map:.4f}")
        print(f"   신규 모델 mAP50 : {new_map:.4f}")

        if new_map >= current_map:
            print("🎉 신규 모델 채택!")
            self.switch_model(engine_path, version, meta)
            self.report_status(version, "active", "Promoted after eval", new_map, meta.get("run_id"))
            self.promote_in_mlflow(version)
        else:
            print("📉 신규 모델 성능 미달 → 기각")
            self.report_status(version, "rejected", "Lower mAP50", new_map, meta.get("run_id"))
            self.demote_in_mlflow(version)

    # ── Golden Set 평가 ──────────────────────────────────────────────────────

    def evaluate_engine(self, engine_path: str) -> float:
        """
        Golden Set으로 엔진을 평가하여 mAP50을 반환합니다.

        Golden Set 준비:
          serving/edge/golden_set/
            images/   ← 고정 검증 이미지 (~100장)
            labels/   ← YOLO 형식 라벨
            golden.yaml

        golden.yaml 예시:
          path: /path/to/serving/edge/golden_set
          train: images
          val:   images
          nc: 6
          names: {0: missing_hole, 1: mouse_bite, ...}

        Golden Set이 없으면 0.0을 반환합니다.
        → 현재 엔진도 0.0이 될 경우 신규 모델은 항상 채택됩니다.
          실제 운영 전에 반드시 golden_set을 준비하세요.
        """
        if not os.path.exists(engine_path):
            return 0.0

        if not os.path.exists(config.GOLDEN_YAML_PATH):
            print(
                "⚠️  Golden Set이 없습니다 (config.GOLDEN_YAML_PATH 미존재).\n"
                "   mAP50=0.0 을 반환합니다. 신규 모델이 항상 채택될 수 있습니다.\n"
                f"   준비 경로: {config.GOLDEN_SET_DIR}"
            )
            return 0.0

        print(f"   🔍 Golden Set 평가 중: {engine_path}")
        try:
            from ultralytics import YOLO
            model = YOLO(engine_path, task='detect')
            metrics = model.val(
                data=config.GOLDEN_YAML_PATH,
                split='val',
                verbose=False,
                device=0
            )
            map50 = float(metrics.box.map50)
            print(f"   ✅ 평가 완료 → mAP50={map50:.4f}")
            return map50
        except Exception as e:
            print(f"   ❌ Golden Set 평가 실패: {e}")
            return 0.0

    # ── 모델 교체 (atomic symlink) ───────────────────────────────────────────

    def switch_model(self, engine_path: str, version, meta=None):
        """
        current.engine 심링크를 atomic하게 교체합니다.

        일반 방식(삭제→생성)은 삭제~생성 사이에 경로가 깨진 상태가
        순간적으로 존재할 수 있습니다. os.rename()은 POSIX에서 atomic이
        보장되므로, 임시 심링크를 먼저 만든 뒤 rename으로 교체합니다.
        """
        target     = config.MODEL_PATH
        tmp_target = target + ".tmp"
        abs_engine = os.path.abspath(engine_path)

        # 임시 심링크 생성 (이미 있으면 삭제 후 재생성)
        if os.path.islink(tmp_target) or os.path.exists(tmp_target):
            os.remove(tmp_target)
        os.symlink(abs_engine, tmp_target)

        # atomic 교체: current.engine이 단 한 순간도 깨진 상태가 되지 않음
        os.rename(tmp_target, target)
        self.current_version = str(version)
        print(f"🔄 심링크 교체 완료: '{target}' → {abs_engine}")

        # MLOps 버전과 YOLO 버전을 하나로 합쳐서 로컬 파일에 저장 (예: yolov11m_v2)
        yolo_version = "yolov11m"
        if meta and "tags" in meta:
            yolo_version = meta["tags"].get("yolo_version", "yolov11m")
            
        combined_model_name = f"{yolo_version}_{self.current_version}"
        version_info = {
            "model_name": combined_model_name
        }
        
        version_file_path = os.path.join(os.path.dirname(config.MODEL_PATH), "current_version.json")
        try:
            import json
            with open(version_file_path, "w") as f:
                json.dump(version_info, f)
            print(f"📄 모델 버전 정보 갱신 완료: {version_file_path}")
        except Exception as e:
            print(f"⚠️ 모델 버전 파일 저장 실패: {e}")

        # inference_worker에 핫스왑 신호 전달
        with open(config.RELOAD_FLAG_PATH, 'w') as f:
            f.write(str(version))
        print(f"🔔 RELOAD_FLAG 생성 → inference_worker가 다음 루프에 모델 교체")

    # ── MLflow 스테이지 승격 ─────────────────────────────────────────────────

    def promote_in_mlflow(self, version_str):
        if not self.client:
            return
        try:
            print(f"🚀 MLflow Registry: v{version_str} → Production 승격 중...")
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=str(version_str),
                stage="Production",
                archive_existing_versions=True
            )
            print(f"✅ MLflow 업데이트 완료: v{version_str} → Production")
        except Exception as e:
            print(f"⚠️  MLflow 스테이지 전환 실패: {e}")
            
    def demote_in_mlflow(self, version_str):
        """성능 미달 모델을 Archived로 전환합니다."""
        if not self.client:
            return
        try:
            print(f"📦 MLflow Registry: v{version_str} → Archived (성능 미달)")
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=str(version_str),
                stage="Archived"
            )
            print(f"✅ MLflow 업데이트 완료: v{version_str} → Archived")
        except Exception as e:
            print(f"⚠️  MLflow Archived 전환 실패: {e}")

    # ── 상태 보고 ────────────────────────────────────────────────────────────

    def report_status(self, version, status, message, map_score=None, run_id=None):
        if not MLFLOW_AVAILABLE:
            return
        try:
            with mlflow.start_run(run_name=f"Edge_Deploy_v{version}"):
                mlflow.log_param("version",  version)
                mlflow.log_param("status",   status)
                mlflow.log_param("message",  message)
                mlflow.set_tag("edge_device",   self.edge_device_id)
                if run_id:
                    mlflow.set_tag("source_run_id", run_id)
                if map_score is not None:
                    mlflow.log_metric("edge_map50", map_score)
            print("📡 상태를 MLflow에 기록했습니다.")
        except Exception as e:
            print(f"⚠️  MLflow 상태 보고 실패: {e}")

    # ── 메인 루프 ────────────────────────────────────────────────────────────

    def run(self):
        print(f"🟢 Updater 시작 (폴링 주기: {self.interval}초)")
        print(f"   조건 필터: {config.REQUIRED_TAG_KEY}={config.REQUIRED_TAG_VALUE!r}")
        print(f"   TRT workspace: {config.TRT_WORKSPACE_MB}MB, avgTiming: {config.TRT_AVG_TIMING_ITERS}")
        try:
            while True:
                # main/inference가 작동 중인지 확인. 바쁨 플래그가 있으면 업데이트를 보류.
                if os.path.exists(config.INFERENCE_BUSY_FLAG_PATH):
                    print("🔒 추론 중이므로 전체 업데이트 사이클 대기...")
                    # 실행 중인 trtexec가 있다면 즉시 종료
                    self._kill_trtexec_if_running()
                    time.sleep(60)  # 1분 후 재확인
                    continue

                self.check_for_updates()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("🟢 Updater 종료됨")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model updater utility")
    parser.add_argument("--abort", action="store_true",
                        help="중단된 TRT 빌드가 있으면 취소하고 관련 파일 삭제")
    parser.add_argument("--status", action="store_true",
                        help="현재 빌드 상태와 플래그 확인")
    args = parser.parse_args()

    updater = ModelUpdater()

    if args.abort:
        updater.abort_build()
    elif args.status:
        updater.print_status()
    else:
        updater.run()
