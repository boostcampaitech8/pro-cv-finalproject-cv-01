"""
QAT 변환 검증 스크립트
"""
import sys
import yaml
from pathlib import Path

# training 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent))

from src.models.yolov8s_qat import get_model

def verify_qat_layers():
    print("=== QAT 변환 검증 시작 ===")
    
    config_path = Path(__file__).parent / 'config_qat.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # 모델 로드 (여기서 QAT 변환이 일어남)
    print("모델 로드 중...")
    model = get_model(config)
    
    print("\n[레이어 타입 확인]")
    conv_found = False
    quant_conv_found = False
    
    # 처음 5개 모듈만 확인
    count = 0
    for name, module in model.model.named_modules():
        name_lower = name.lower()
        if 'conv' in name_lower and not 'bottleneck' in name_lower:
            mod_type = type(module).__name__
            print(f"- {name}: {mod_type}")
            
            if 'QuantConv2d' in mod_type:
                quant_conv_found = True
            elif 'Conv2d' in mod_type:
                 # QuantConv2d는 Conv2d를 상속받을 수 있으므로 순서 중요
                 if not 'Quant' in mod_type: 
                    conv_found = True
            
            count += 1
            if count >= 10: 
                break
                
    print("\n[검증 결과]")
    if quant_conv_found:
        print("✅ 성공: QuantConv2d 레이어가 발견되었습니다.")
        print("   QAT가 정상적으로 적용될 준비가 되었습니다.")
    else:
        print("❌ 실패: QuantConv2d 레이어가 발견되지 않았습니다.")
        print("   initialize_quantization() 호출을 확인하세요.")

if __name__ == "__main__":
    verify_qat_layers()
