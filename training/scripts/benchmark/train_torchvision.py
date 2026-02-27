import os
import argparse
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.rpn import AnchorGenerator
from ultralytics.utils import SimpleNamespace

# src 및 상위 디렉토리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datasets.torchvision_adapter import PCBTorchvisionDataset, collate_fn
from utils.torchvision_wrapper import TorchvisionModelWrapper
from utils.torchvision_validator import TorchvisionValidator
from torchvision.models.detection.rpn import AnchorGenerator

def get_model(model_name, num_classes, imgsz=640):
    # Torchvision은 배경(Index 0)을 포함한 클래스 수가 필요합니다.
    # 데이터셋에는 N개의 클래스가 있으므로 N+1을 전달해야 합니다.
    if model_name == 'fasterrcnn':
        # YOLO의 letterbox 스케일과 맞추기 위해 긴 변(imgsz)을 기준으로 설정합니다.
        # min_size는 짧은 변이 적절히 줄어들도록 설정합니다.
        # 예: imgsz=640일 때 min_size=400 정도가 적절함.
        min_sz = int(imgsz * 0.625) # approx 400 for 640
        
        # 1. 작은 객체 탐지를 위한 커스텀 앵커 (PCB 결함 탐지에 중요)
        # 기본값: ((32,), (64,), (128,), (256,), (512,))
        # 작은 결함을 위해 한 단계 낮춰서 설정: ((8,), (16,), (32,), (64,), (128,))
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        # 2. Backbone Unfreeze (COCO 도메인이 아니므로 전체 미세조정 필요)
        model = fasterrcnn_resnet50_fpn(
            weights='DEFAULT', 
            min_size=min_sz, 
            max_size=imgsz,
            trainable_backbone_layers=5, # 모든 레이어 학습 가능
            rpn_anchor_generator=rpn_anchor_generator
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
    elif model_name == 'ssd':
        model = ssd300_vgg16(weights='DEFAULT', trainable_backbone_layers=5) # VGG 전체 학습 가능
        # Head 교체
        in_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head = SSDHead(in_channels, num_anchors, num_classes + 1)
        
    elif model_name == 'ssd512':
        # SSD300 구조를 사용하되 512 입력을 강제함.
        # 주의: 앵커 크기도 비례해서 커짐 (0.1 * 512 = 51px vs 30px). 
        # 작은 결함에는 불리할 수 있으나 고해상도 특징 맵 활용 가능.
        model = ssd300_vgg16(weights='DEFAULT', trainable_backbone_layers=5)
        # Transform을 512로 강제 변경
        model.transform.min_size = (512,)
        model.transform.max_size = 512
        
        in_channels = det_utils.retrieve_out_channels(model.backbone, (512, 512))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head = SSDHead(in_channels, num_anchors, num_classes + 1)

    elif model_name == 'ssdlite':
        # SSD Lite (MobileNetV3) - 엣지 디바이스 최적화
        # 기본 내부 입력 크기는 320x320
        model = ssdlite320_mobilenet_v3_large(weights='DEFAULT', trainable_backbone_layers=6) # MobileNet(6 layers) 전체 학습
        # Head 교체
        in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        
        import functools
        norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head = SSDLiteHead(in_channels, num_anchors, num_classes + 1, norm_layer=norm_layer)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# ... (omitted lines) ...

    # Log to CSV
    try:
        from benchmark_utils import log_to_csv
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from benchmark_utils import log_to_csv

    try:
        # Extract Class metrics
        # Validator returns 'metrics/mAP50-95(B)' generally.
        class_aps = test_validator.metrics.maps # Array of mAP50-95
        
        log_data = {
            'Model': args.model,
            'Resolution': args.imgsz,
            'Patience': args.patience,
            'mAP50': map50,
            'mAP50-95': map50_95,
            'Speed(ms)': total_time_ms,
            'FPS': fps
        }
        
        # names mapping
        # Ensure we don't crash if len(names) != len(class_aps)
        for i, ap in enumerate(class_aps):
             if i in names:
                cls_name = names[i] + '_AP' 
                log_data[cls_name] = ap
             else:
                print(f"Warning: Class index {i} not in names dict.")
        
        log_to_csv(log_data)
    except Exception as e:
        print(f"ERROR: Failed to log to CSV: {e}")
        import traceback
        traceback.print_exc()


def train(args):
    # 1. 설정 (Config)
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    nc = len(data_cfg['names'])
    names = data_cfg['names'] # dict {0: 'name'}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Benchmark] 학습 시작: {args.model} (디바이스: {device}, 클래스 수: {nc})")

    # 2. 모델 (Model)
    model = get_model(args.model, nc, args.imgsz)
    model.to(device)
    model.to(device)

    # 3. 데이터셋 (학습용) - 공정한 비교를 위해 Ultralytics DataLoader 사용 (Mosaic/MixUp 적용)
    from ultralytics.data import build_dataloader, YOLODataset
    try:
        from ultralytics.cfg import get_cfg # Newer versions
    except ImportError:
        from ultralytics.utils import get_cfg # Older versions? Or ultralytics.yolo.cfg? Check utils.
    
    from ultralytics.utils import DEFAULT_CFG
    
    def get_yolo_dataloader(data_path, imgsz, batch, workers):
        # We need a cfg object or dict. 
        # Ultralytics build_dataloader expects `args` (SimpleNamespace or dict) 
        # that has 'data', 'imgsz', 'batch', etc.
        # But wait, build_dataloader takes (dataset, batch, workers, shuffle, rank).
        # We need YOLODataset first.
        
        # Load data.yaml to get path info? 
        # Actually YOLODataset needs 'data' arg (path to yaml).
        # And 'imgsz'.
        # And 'augment=True'. (This enables Mosaic/MixUp).
        
        # Let's use build_yolo_dataset if available or just YOLODataset.
        # YOLODataset signature: (img_path, imgsz=640, batch_size=16, augment=False, hyp=None, rect=False, cache=False, single_cls=False, stride=32, pad=0.0, min_items=0, classes=None, data=None, task='detect', fraction=1.0, name=None)
        
        # We need hyp.
        # Default hyp from DEFAULT_CFG?
        hyp = DEFAULT_CFG
        
        # Ultralytics YOLODataset 초기화
        dataset = YOLODataset(
            img_path=data_cfg['train'], # train.txt 경로
            imgsz=imgsz,
            batch_size=batch,
            augment=True, # Mosaic/MixUp 활성화
            hyp=hyp, # 기본 하이퍼파라미터 사용
            rect=False, # Mosaic를 위해 정사각형 학습
            cache=False,
            stride=32,
            data=data_cfg # 데이터 설정 전달
        )
        
        loader = build_dataloader(dataset, batch=batch, workers=workers, shuffle=True, rank=-1)
        return loader

    def yolo_to_torchvision(batch, device):
        # YOLO 배치 데이터를 Torchvision 입력 형식으로 변환
        # batch['img']: (B, 3, H, W) 텐서, 0-1 정규화됨
        # batch['cls']: (N, 1) 텐서
        # batch['bboxes']: (N, 4) 텐서, xywh 정규화됨
        # batch['batch_idx']: (N, 1) 텐서
        
        images = batch['img'].to(device) # (B, 3, H, W)
        
        # dtype 확인 (uint8 수정)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
            
        # 타겟을 디바이스로 이동하여 RuntimeError 방지
        batch_cls = batch['cls'].to(device)
        batch_bboxes = batch['bboxes'].to(device)
        batch_idx = batch['batch_idx'].to(device)

        # Torchvision은 List[Tensor] 형식을 기대함
        image_list = [img for img in images]
        
        targets = []
        bs = len(images)
        h, w = images.shape[2], images.shape[3]
        
        for i in range(bs):
            idx = (batch_idx == i).squeeze()
            if idx.sum() == 0:
                # 타겟 없음
                boxes = torch.zeros((0, 4), device=device)
                labels = torch.zeros((0,), dtype=torch.int64, device=device)
            else:
                # 추출
                this_cls = batch_cls[idx].view(-1)
                this_bboxes = batch_bboxes[idx].view(-1, 4)
                
                # xywhn -> xyxy 절대좌표 변환
                # xywhn: cx, cy, w, h (0-1)
                cx, cy, bw, bh = this_bboxes.unbind(1)
                x1 = (cx - 0.5 * bw) * w
                y1 = (cy - 0.5 * bh) * h
                x2 = (cx + 0.5 * bw) * w
                y2 = (cy + 0.5 * bh) * h
                
                boxes = torch.stack([x1, y1, x2, y2], dim=1)
                
                # 라벨: YOLO 0-based -> TV 1-based (0은 배경)
                labels = (this_cls + 1).long()
            
            target = {
                "boxes": boxes,
                "labels": labels
            }
            targets.append(target)
            
        return image_list, targets

    train_loader = get_yolo_dataloader(args.data, args.imgsz, args.batch, workers=4)

    # 4. 옵티마이저 및 스케줄러 (Optimizer & Scheduler)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 스케줄러: Cosine Annealing 사용 요청됨.
    # StepLR보다 부드럽게 감소하여 일반적으로 더 좋은 성능을 보임.
    # T_max = 전체 에포크. eta_min = 종료 시 최소 LR.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00005)

    # 체크포인트에서 재개 (있는 경우)
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"{args.resume} 에서 학습을 재개합니다.")
        checkpoint = torch.load(args.resume)
        # 모델 가중치 로드
        model.load_state_dict(checkpoint)
        # 옵티마이저와 스케줄러 상태도 로드하는 것이 좋으나 저장하지 않았음.
        # 미세조정(Fine-tuning) 단계에서는 옵티마이저를 초기화해도 괜찮음.
        pass

    # 5. 학습 루프 (Training Loop)
    save_dir = f"runs/benchmark/{args.model}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 6. 조기 종료 설정 (Early Stopping)
    best_fitness = 0.0
    patience_counter = 0
    
    from pathlib import Path
    
    # 검증 설정 (Validation Setup)
    val_args = SimpleNamespace(
        task='detect', mode='val', data=args.data, imgsz=args.imgsz, 
        batch=args.batch, conf=0.001, iou=0.6, device=str(device), 
        half=False, save_dir=Path(save_dir) / 'val', plots=False
    )
    
    # 검증기 초기화 (한 번만 수행)
    validator = TorchvisionValidator(args=val_args, save_dir=val_args.save_dir)
    
    for epoch in range(args.epochs):
        # --- 학습 (Training) ---
        model.train()
        torch.set_grad_enabled(True)
        # 모든 파라미터 Unfreeze (검증기가 Freeze 시켰을 수 있음)
        for param in model.parameters():
            param.requires_grad = True
            
        total_loss = 0
        
        # New Loop for YOLO Loader
        # pbar? Ultralytics loader is just an iterator?
        # Typically we wrap it in tqdm if we want.
        # But let's keep it simple.
        
        # for i, batch in enumerate(train_loader):
        # We need to handle the fact that train_loader might restart or be infinite?
        # Ultralytics loader is finite (based on dataset length).
        
        for i, batch in enumerate(train_loader):
            # Convert
            images, targets = yolo_to_torchvision(batch, device)
            
            # images is List[Tensor], targets is List[Dict]
            # No need to .to(device) again as yolo_to_torchvision does it.

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # NaN/Inf 확인
            if not torch.isfinite(losses):
                print(f"경고: 손실값이 무한대이거나 NaN입니다 {losses}, 스텝을 건너뜁니다.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) # Gradient Clipping
            optimizer.step()
            
            total_loss += losses.item()
            
        # Scheduler Step
        lr_scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # --- 검증 및 조기 종료 (Validation & Early Stopping) ---
        # 검증을 위한 모델 래퍼
        wrapper = TorchvisionModelWrapper(model)
        wrapper.names = names
        wrapper.to(device)
        wrapper.eval()
        
        # 검증 실행
        # 미리 초기화된 검증기 사용
        stats = validator(model=wrapper)
        
        # Fitness 확인 (mAP50)
        current_fitness = stats['metrics/mAP50(B)']
        print(f"[검증] mAP50: {current_fitness:.4f}")

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            patience_counter = 0
            # 최고 성능 모델 저장
            torch.save(model.state_dict(), f"{save_dir}/best.pt")
            print(f"새로운 최고 성능 모델 저장됨! ({best_fitness:.4f})")
        else:
            patience_counter += 1
            print(f"성능 향상 없음. 인내심: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print("조기 종료(Early Stopping) 발동.")
            break

    # 7. 최종 테스트 평가 (Final Test Evaluation)
    print("\n[Benchmark] 테스트(TEST) 세트에서 최종 평가 시작...")
    
    # 최고 성능 웨이트 로드
    best_weights = f"{save_dir}/best.pt"
    if os.path.exists(best_weights):
        model.load_state_dict(torch.load(best_weights))
        print(f"최고 성능 웨이트 로드됨: {best_weights}")
    
    wrapper = TorchvisionModelWrapper(model)
    wrapper.names = names
    wrapper.to(device)
    wrapper.eval()
    
    # Test Args
    test_args = SimpleNamespace(
        task='detect', mode='val', split='test', data=args.data, 
        imgsz=args.imgsz, batch=args.batch, conf=0.001, iou=0.6, 
        device=str(device), half=False, save_dir=Path(save_dir) / 'test', plots=True
    )
    
    test_validator = TorchvisionValidator(args=test_args, save_dir=test_args.save_dir)
    test_stats = test_validator(model=wrapper)
    
    # FPS 계산
    # Validator는 속도를 self.speed 딕셔너리에 저장함
    speed = test_validator.speed
    total_time_ms = speed['preprocess'] + speed['inference'] + speed['loss'] + speed['postprocess']
    fps = 1000.0 / total_time_ms

    print(f"==================================================")
    print(f"[{args.model}] Final Test Metircs:")
    # Debug: Print available keys
    print(f"Available stats keys: {list(test_stats.keys())}")
    
    map50 = test_stats.get('metrics/mAP50(B)', 0.0)
    map50_95 = test_stats.get('metrics/mAP50-95(B)', 0.0)
    
    print(f" >> mAP50    : {map50:.4f}")
    print(f" >> mAP50-95 : {map50_95:.4f}")
    print(f" >> Speed    : {total_time_ms:.2f}ms/img ({fps:.2f} FPS)")
    print(f"==================================================")

    # Log to CSV
    try:
        from benchmark_utils import log_to_csv
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from benchmark_utils import log_to_csv

    try:
        # Extract Class metrics
        # Validator returns 'metrics/mAP50-95(B)' generally.
        class_aps = test_validator.metrics.maps # Array of mAP50-95
        
        log_data = {
            'Model': args.model,
            'Resolution': args.imgsz,
            'Patience': args.patience,
            'mAP50': map50,
            'mAP50-95': map50_95,
            'Speed(ms)': total_time_ms,
            'FPS': fps
        }
        
        # names mapping
        # Ensure we don't crash if len(names) != len(class_aps)
        for i, ap in enumerate(class_aps):
             if i in names:
                cls_name = names[i] + '_AP' 
                log_data[cls_name] = ap
             else:
                print(f"Warning: Class index {i} not in names dict.")
        
        log_to_csv(log_data)
    except Exception as e:
        print(f"오류: CSV 로그 저장 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Torchvision 모델 벤치마크 학습 스크립트")
    parser.add_argument('--model', type=str, default='fasterrcnn', choices=['fasterrcnn', 'ssd', 'ssdlite', 'ssd512'], help="학습할 모델 선택")
    parser.add_argument('--data', type=str, default='PCB_DATASET/data.yaml', help="데이터셋 YAML 파일 경로")
    parser.add_argument('--epochs', type=int, default=50, help="학습 에포크 수") 
    parser.add_argument('--batch', type=int, default=4, help="배치 사이즈")
    parser.add_argument('--patience', type=int, default=10, help="조기 종료 인내심 (Early Stopping Patience)")
    parser.add_argument('--imgsz', type=int, default=640, help="입력 이미지 크기")
    parser.add_argument('--resume', type=str, default='', help='학습을 재개할 체크포인트 경로')
    args = parser.parse_args()
    
    train(args)
