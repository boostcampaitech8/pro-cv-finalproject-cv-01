import subprocess
import time
import sys
import os

def run_command(cmd):
    """
    명령어를 실행하고 성공 여부를 반환합니다.
    """
    print(f"\n[Benchmark] 실행 중: {cmd}")
    start_time = time.time()
    try:
        # 쉘을 통해 명령어 실행 (bash 사용)
        result = subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
        duration = time.time() - start_time
        print(f"[Benchmark] 완료 소요 시간: {duration:.1f}초")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[Benchmark] 실패: {cmd}")
        print(f"에러 코드: {e.returncode}")
        return False

def main():
    # 설정 (Configuration)
    # 전체 실행 또는 빠른 테스트를 위해 에포크(epochs)와 인내심(patience)을 조정하세요.
    # 전체 실행 권장값: RT-DETR=100~150, Torchvision=50~100, Patience=10~20
    RT_EPOCHS = 150
    TV_EPOCHS = 100
    RT_PATIENCE = 20
    TV_PATIENCE = 10
    
    # 배치 사이즈 (고해상도 학습을 위해 보수적으로 설정)
    BATCH_SIZE = 4 
    
    # 스크립트 경로 설정 (현재 파일 위치 기준)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_torchvision.py")
    
    # 실행할 실험 명령어 목록
    commands = [
        # 1. Faster R-CNN (640 & 1280 해상도)
        f"python {TRAIN_SCRIPT} --model fasterrcnn --imgsz 640 --epochs {TV_EPOCHS} --batch 8 --patience {TV_PATIENCE}",
        f"python {TRAIN_SCRIPT} --model fasterrcnn --imgsz 1280 --epochs {TV_EPOCHS} --batch 2 --patience {TV_PATIENCE}", # 무거운 백본 사용 시 배치 사이즈 감소
        
        # 2. SSD 300 (VGG16) - 내부 입력 크기 300 고정
        f"python {TRAIN_SCRIPT} --model ssd --imgsz 640 --epochs {TV_EPOCHS} --batch 8 --patience {TV_PATIENCE}",
        
        # 3. SSD Lite (MobileNetV3) - 내부 입력 크기 320 고정 (엣지 디바이스 최적화)
        f"python {TRAIN_SCRIPT} --model ssdlite --imgsz 640 --epochs {TV_EPOCHS} --batch 16 --patience {TV_PATIENCE}",

        # 4. SSD 512 (VGG16 Scaled) - 내부 입력 크기 512 고정
        f"python {TRAIN_SCRIPT} --model ssd512 --imgsz 512 --epochs {TV_EPOCHS} --batch 4 --patience {TV_PATIENCE}",
    ]
    
    print("==================================================")
    print(f"벤치마크 스위트 시작: 총 {len(commands)}개의 실험")
    print("결과는 다음 파일에 저장됩니다: runs/benchmark_results.csv")
    print("==================================================")
    
    failures = []
    
    for i, cmd in enumerate(commands):
        print(f"\n>> 실험 {i+1}/{len(commands)}")
        success = run_command(cmd)
        if not success:
            failures.append(cmd)
            
    print("\n==================================================")
    print("벤치마크 스위트 종료.")
    
    if failures:
        print(f"검증 경고: {len(failures)}개의 실험이 실패했습니다:")
        for f in failures:
            print(f" - {f}")
    else:
        print("모든 실험이 성공적으로 완료되었습니다!")
        
    print("runs/benchmark_results.csv 파일을 확인하세요.")
    print("==================================================")

if __name__ == "__main__":
    main()
