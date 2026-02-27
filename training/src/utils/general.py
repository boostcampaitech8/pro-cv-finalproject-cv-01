import os
import shutil
import sys
import glob
import re
from pathlib import Path

def cleanup_artifacts(save_dir, config):
    """
    모든 훈련이 완료된 후, 동떨어진 임시 디렉토리 및 쓰레기 파일들을 검사 후 일괄 청소합니다.
    'runs/detect' 위치 폴더를 최종 저장 경로인 '{save_dir}/detect' 로 병합 이동 시키고 자동 혼합 정밀도(AMP) 체크용 폐기 파일들을 영구 삭제합니다.
    """
    print("\n최종 아티팩트 청소(Cleanup) 로직을 실행합니다...")
    
    # 예측 및 탐지 결과물들이 병합될 타겟 디렉토리 경로 정의
    target_detect = os.path.join(save_dir, "detect")
    
    # 제거가 필요한 임시 부산물 대상 리스트업
    # 1. 'runs/detect' (간혹 Ultralytics 시스템 기본 매커니즘에 의해 의도치 않게 생성됨)
    stray_detect_runs = os.path.join("runs", "detect")
    # 2. 'detect' (프로젝트 절대경로 생성이 실패했을 시 종종 루트 환경에 최상위 접근으로 생김)
    stray_detect_root = "detect"
    # 3. AMP 기능 진단 유무용 모델 아티팩트 찌꺼기들
    stray_amp_model_11 = "yolo11n.pt"
    stray_amp_model_8 = "yolov8n.pt"

    strays = [stray_detect_runs, stray_detect_root, stray_amp_model_11, stray_amp_model_8]

    for stray in strays:
        if os.path.exists(stray):
            # Case A: 독립적인 파일 구조인 경우 (AMP 결과 파일 등) -> 영구 삭제
            if os.path.isfile(stray):
                try:
                    os.remove(stray)
                    print(f"Cleanup: '{stray}' 폐기 임시 파일을 성공적으로 제거를 완료했습니다 (AMP check 아티팩트).")
                except Exception as e:
                    print(f"'{stray}' 삭제에 실패했습니다 사유: {e}")
                continue

            # Case B: 디렉토리(폴더) 구조인 경우 (임시 잔여 보관 결괏값) -> 타겟 방향으로 이관 및 합산 처리
            print(f"알림: 잔여 디렉토리 파편인 '{stray}' 이(가) 발견되었습니다. {target_detect} 방향으로 경로를 병합 이도합니다...")
            try:
                os.makedirs(save_dir, exist_ok=True)
                
                # 폴더 이동 전 타겟 위치 존재 유무 파악
                if not os.path.exists(target_detect):
                    shutil.move(stray, target_detect)
                else:
                    # 해당 타겟 대상지가 이미 온전한 경우 파일/하위폴더별 단일 융합 처리 로직 적용
                    for item in os.listdir(stray):
                        s = os.path.join(stray, item)
                        d = os.path.join(target_detect, item)
                        if os.path.exists(d):
                            if os.path.isdir(d): shutil.rmtree(d)
                            else: os.remove(d)
                        shutil.move(s, d)
                    shutil.rmtree(stray)
                print(f"'{stray}' 병합 이전 절차를 모두 성공리에 마쳤습니다.")
            except Exception as cleanup_e:
                print(f"'{stray}' 이동 작업이 반려되어 실패했습니다: {cleanup_e}")

def set_seed(seed=42):
    """
    모든 기계 학습 연산의 재현성을 통제 가능 수준으로 보장하기 위하여 각종 난수 시드(Seed) 값들을 일률적으로 고정합니다.
    (참고: 본 기능과 무관하게 YOLOv8 자체 내장 시스템 내에서 훈련용 인자 'seed'를 통해 독립적으로 처리하지만, 
     타 여타 라이브러리 및 불특정 무작위 연산을 모두 완벽히 무력화하기에 권장되는 조치입니다.)
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class Tee(object):
    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    
    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush() # 시스템 장애를 대비하기 위한 매우 즉각적인 데이터 I/O 배출 보장 강제화
        self.stdout.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # 단일 명칭으로만 지정되어있는 디렉토리 및 파일 이름 끝쪽에 숫자를 덧붙혀 무단 덮어씌움 사고를 봉쇄합니다. 
    # 즉,  runs/exp 접근 시 -> runs/exp{sep}2, runs/exp{sep}3 순차증가 증분으로 재매핑
    path = Path(path)  # OS 시스템 독립적 호환 객체
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        
        # 루프를 돌면서 여백 탐색 증분 검사 (기법 1)
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # 지정된 이름의 재결합
            if not os.path.exists(p):  # 만약 미점유 상태인 경우 확정
                path = Path(p)
                break
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # 중간 매개체 모두를 포함한 디렉터리 동시 확립 생성 보장
        
    return path

def setup_logging(save_dir):
    """
    stdout(표준 출력) 및 stderr(표준 에러 로그)의 실시간 노출 방향을 파일인 'console.log'에 동시에 기록하도록 Tee 로깅 객체를 접합시켜 주입합니다.
    """
    log_path = os.path.join(save_dir, "console.log")
    # 프로세스 스레딩이 복수로 진행되거나, 핫클래스 호출(reload)이 일어나 동일 자원에 로깅 파이프가 다중 장착되는 중첩 결함 현상 회피 방어 코드
    if not isinstance(sys.stdout, Tee):
        Tee(log_path)
        print(f"로깅 객체가 연동 성공했습니다. 모든 커맨드 라인 내역은 {log_path} 파일에도 항시 백업됩니다.")
