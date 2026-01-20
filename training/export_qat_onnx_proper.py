#!/usr/bin/env python3
"""
QAT 모델 ONNX Export - Q/DQ 노드 포함 보장

pytorch-quantization을 직접 사용하여 Q/DQ 노드를 포함한 ONNX 생성
"""

import torch
import sys
from pathlib import Path


def export_with_qdq_nodes(model_path: str, output_path: str = None):
    """
    Q/DQ 노드를 포함한 ONNX export

    Args:
        model_path: .pt 파일 경로
        output_path: .onnx 저장 경로
    """
    model_path = Path(model_path)

    if output_path is None:
        output_path = str(model_path).replace('.pt', '_qat.onnx')

    print("="*60)
    print("QAT 모델 → ONNX Export (Q/DQ 노드 포함)")
    print("="*60)
    print(f"입력: {model_path}")
    print(f"출력: {output_path}")
    print("="*60)

    # 1. pytorch-quantization 확인
    print("\n[1/5] pytorch-quantization 확인...")
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization import calib
        print("  ✅ pytorch-quantization 설치됨")
    except ImportError:
        print("  ❌ pytorch-quantization 없음")
        print("\n설치 필요:")
        print("  pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")
        return None

    # 2. 모델 로드
    print("\n[2/5] 모델 로드...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  - 디바이스: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 모델 추출
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'ema' in checkpoint:
            model = checkpoint['ema']
        else:
            model = checkpoint
    else:
        model = checkpoint

    model = model.to(device)
    model.eval()
    print("  ✅ 모델 로드 완료")

    # 3. Q/DQ 노드 활성화 확인
    print("\n[3/5] Q/DQ 노드 확인...")
    has_quantizers = False
    quantizer_count = 0

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            has_quantizers = True
            quantizer_count += 1

    print(f"  - TensorQuantizer 모듈 개수: {quantizer_count}")

    if not has_quantizers:
        print("  ⚠️ 경고: TensorQuantizer가 없습니다.")
        print("  이 모델은 QAT 학습되지 않았을 수 있습니다.")
    else:
        print("  ✅ QAT 모델 확인됨")

    # 4. ONNX Export
    print("\n[4/5] ONNX Export 시작...")

    # Fake quantization 모드 활성화 (중요!)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    print("  - Fake quantization 모드 활성화")

    # Dummy input
    dummy_input = torch.randn(1, 3, 640, 640, device=device)

    # 여러 방법 시도
    success = False

    # 방법 1: 낮은 opset + older exporter
    for opset in [11, 10, 9]:
        try:
            print(f"\n  [방법 1] Opset {opset} + older exporter 시도...")

            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output0'],
                dynamic_axes=None,
                verbose=False,
                # Older exporter 사용 (dynamo 비활성화)
                dynamo=False,
            )

            print(f"  ✅ Opset {opset} 성공!")
            success = True
            break

        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ 실패: {error_msg[:80]}...")

            # dynamo 파라미터가 없는 경우 시도
            if "unexpected keyword argument 'dynamo'" in error_msg:
                try:
                    print(f"  [재시도] dynamo 없이...")
                    torch.onnx.export(
                        model,
                        dummy_input,
                        output_path,
                        export_params=True,
                        opset_version=opset,
                        do_constant_folding=True,
                        input_names=['images'],
                        output_names=['output0'],
                        dynamic_axes=None,
                        verbose=False,
                    )
                    print(f"  ✅ Opset {opset} 성공!")
                    success = True
                    break
                except Exception as e2:
                    print(f"  ❌ 재시도 실패: {str(e2)[:80]}...")

            continue

    if not success:
        print("\n  ⚠️ torch.onnx.export 실패")
        quant_nn.TensorQuantizer.use_fb_fake_quant = False
        return None

    # 설정 복원
    quant_nn.TensorQuantizer.use_fb_fake_quant = False

    # 5. 검증
    print("\n[5/5] ONNX 검증 및 Q/DQ 노드 확인...")
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ✅ ONNX 모델 검증 통과")

        # 파일 크기
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"  - 파일 크기: {file_size:.1f} MB")

        # Q/DQ 노드 확인 (핵심!)
        qdq_count = {}
        for node in onnx_model.graph.node:
            if node.op_type in ['QuantizeLinear', 'DequantizeLinear']:
                qdq_count[node.op_type] = qdq_count.get(node.op_type, 0) + 1

        print("\n" + "="*60)
        if qdq_count:
            print("🎉 성공! QAT Q/DQ 노드 발견!")
            for op, count in qdq_count.items():
                print(f"  - {op}: {count}개")
            print("\n✅ 엣지에서 calibration 불필요!")
            print("✅ 이 파일을 엣지 담당자에게 전달하세요!")
        else:
            print("⚠️ Q/DQ 노드가 발견되지 않았습니다.")
            print("\n가능한 원인:")
            print("1. 모델이 QAT 학습되지 않음")
            print("2. ONNX export 시 Q/DQ 노드가 최적화로 제거됨")
            print("3. pytorch-quantization 버전 호환성 문제")
            print("\n→ 이 파일도 사용 가능하지만 엣지에서 calibration 필요")
        print("="*60)

    except Exception as e:
        print(f"  ⚠️ 검증 실패: {e}")

    return output_path


def main():
    import sys

    if len(sys.argv) < 2:
        print("사용법: python export_qat_onnx_proper.py <model.pt> [output.onnx]")
        print("\n예시:")
        print("  python export_qat_onnx_proper.py runs/qat/qat_yolov8s2/weights/best.pt")
        print("  python export_qat_onnx_proper.py best.pt best_qat.onnx")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    result = export_with_qdq_nodes(model_path, output_path)

    if result:
        print(f"\n✅ 완료! ONNX 파일: {result}")
        print("\n다음 단계:")
        print("1. 엣지 담당자에게 이 파일 전달")
        print("2. Jetson에서 TensorRT 변환:")
        print(f"   trtexec --onnx={Path(result).name} \\")
        print("           --saveEngine=model.engine \\")
        print("           --int8 --workspace=4096 --fp16")
    else:
        print("\n❌ ONNX export 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
