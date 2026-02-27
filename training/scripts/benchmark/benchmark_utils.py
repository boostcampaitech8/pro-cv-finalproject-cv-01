import csv
import os
import datetime

def log_to_csv(results, filepath='runs/benchmark_results.csv'):
    """
    벤치마크 결과를 CSV 파일에 저장합니다.
    
    Args:
        results (dict): 저장할 메트릭 데이터 (Model, Resolution, mAP50, Speed 등 포함)
        filepath (str): CSV 파일 경로 (기본값: runs/benchmark_results.csv)
    """
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 타임스탬프 추가
    results['Timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # CSV 컬럼 순서 정의
    fieldnames = [
        'Timestamp', 'Model', 'Resolution', 'Patience', 
        'mAP50', 'mAP50-95', 'Speed(ms)', 'FPS',
        'missing_hole_AP', 'mouse_bite_AP', 'open_circuit_AP', 
        'short_AP', 'spur_AP', 'spurious_copper_AP'
    ]
    
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as f:
        # extrasaction='ignore'는 fieldnames에 없는 키가 있어도 에러를 발생시키지 않도록 함
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        
        # 파일이 새로 생성되었으면 헤더 작성
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(results)
    
    print(f"[Benchmark] 결과가 {filepath} 에 추가되었습니다.")
