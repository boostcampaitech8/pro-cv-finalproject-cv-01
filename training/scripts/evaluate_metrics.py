import argparse
import os
import time
import glob
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# 한글 출력을 위한 유니코드 처리 (필요시)
import sys

# ==============================================================================
# Helper Classes (Pickling Safe)
# ==============================================================================
class FeatureHook:
    """
    Hook to capture feature maps or logits. 
    Must be present during unpickling if the model was saved with hooks.
    """
    def __init__(self, storage, idx):
        self.storage = storage
        self.idx = idx

    def __call__(self, module, input, output, **kwargs):
        self.storage[self.idx] = output

class FeatureAdapter(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.adapters = torch.nn.ModuleList()

class KDLoss:
    pass

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes (x, y, w, h) normalized.
    box1: [x_c, y_c, w, h]
    box2: [x_c, y_c, w, h]
    """
    # Convert to x1, y1, x2, y2
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    
    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    
    inter_area = inter_w * inter_h
    
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    
    union_area = b1_area + b2_area - inter_area + 1e-16
    
    return inter_area / union_area

def load_gt(label_path):
    """
    Load ground truth from a text file.
    Returns list of [class_id, x, y, w, h]
    """
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    boxes.append(parts) # cls, x, y, w, h
    return boxes

def evaluate_metrics(weights, data_config, test_images_txt, imgsz=640, conf_thres=0.25, iou_thres=0.5, device='0'):
    print(f"\n{'='*20} 종합 메트릭 평가 시작 {'='*20}")
    print(f"모델: {weights}")
    print(f"데이터 설정: {data_config}")
    
    # 1. 모델 로드
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 2. 표준 mAP 평가 (YOLO Val 사용)
    print("\n[1/3] 표준 mAP 평가 중...")
    try:
        # data_config 파일을 절대 경로로 변환
        abs_data_path = os.path.abspath(data_config)
        
        # Verify the yaml has 'test' key if we want to evaluate on test set
        with open(abs_data_path, 'r') as f:
            d = yaml.safe_load(f)
            if 'test' not in d:
                 print(f"  -> 경고: {abs_data_path}에 'test' 경로가 없습니다. 'val' 데이터를 대체 사용합니다.")
                 split_to_use = 'val'
            else:
                 split_to_use = 'test'
            
        print(f"  -> Using data path: {abs_data_path} (split: {split_to_use})")
        val_results = model.val(data=abs_data_path, split=split_to_use, imgsz=imgsz, device=device, verbose=False)
        map50 = val_results.box.map50
        map50_95 = val_results.box.map
        
        print(f"  -> mAP50    : {map50:.4f}")
        print(f"  -> mAP50-95 : {map50_95:.4f}")
    except Exception as e:
        print(f"  -> 표준 평가 실패 (이 단계는 건너뜁니다): {e}")
        map50, map50_95 = 0.0, 0.0

    # 3. FPS 및 Latency 측정
    print("\n[2/3] FPS 및 레이턴시 측정 중...")
    
    # 웜업
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(source=dummy_input, imgsz=imgsz, device=device, verbose=False, half=False)
        
    # 실제 측정 (테스트 이미지 중 일부 사용)
    with open(test_images_txt, 'r') as f:
        img_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(img_paths) > 100:
        measure_paths = img_paths[:100] # 속도를 위해 100장만
    else:
        measure_paths = img_paths
        
    latencies = []
    pre_times = []
    inf_times = []
    post_times = []
    
    for img_path in tqdm(measure_paths, desc="속도 측정"):
        res = model.predict(source=img_path, imgsz=imgsz, device=device, verbose=False, half=False)
        speed = res[0].speed
        pre, inf, post = speed['preprocess'], speed['inference'], speed['postprocess']
        total = pre + inf + post
        latencies.append(total)
        pre_times.append(pre)
        inf_times.append(inf)
        post_times.append(post)
        
    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    print(f"  -> 평균 레이턴시 : {avg_latency:.2f} ms")
    print(f"     (전처리: {np.mean(pre_times):.2f} ms, 추론: {np.mean(inf_times):.2f} ms, 후처리: {np.mean(post_times):.2f} ms)")
    print(f"  -> FPS          : {fps:.2f} FPS")
    
    # 4. 세부 Recall / Precision 계산
    print("\n[3/3] 세부 Recall 및 Precision 계산 중 (이미지별/클래스별)...")
    
    # 클래스 이름 로드
    with open(data_config, 'r') as f:
        data_yaml = yaml.safe_load(f)
    try:
        class_names = data_yaml['names']
        # 딕셔너리인 경우 리스트로 변환
        if isinstance(class_names, dict):
            class_names = [class_names[i] for i in sorted(class_names.keys())]
    except:
        class_names = val_results.names # Fallback

    # 통계 저장소
    # 클래스별: {cls_id: {'TP': 0, 'FP': 0, 'FN': 0}}
    class_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    
    # 이미지별 결과 저장
    image_results = [] # {'image': name, 'precision': p, 'recall': r}
    
    for img_path in tqdm(img_paths, desc="정밀도 평가"):
        # 1. 추론
        res = model.predict(source=img_path, imgsz=imgsz, device=device, verbose=False, half=False)[0]
        
        # 예측 박스 파싱: [cls, conf, x, y, w, h] (normalized)
        preds = []
        for box in res.boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            if conf < conf_thres:
                continue
            # xywhn (정규화된 중심좌표 w h)
            x, y, w, h = box.xywhn[0].tolist()
            preds.append({'cls': cls_id, 'box': [x, y, w, h], 'matched': False})
            
        # 2. GT 로드 (라벨 파일 경로 유추)
        # images/test/image.jpg -> labels/test/image.txt
        # dataset.py의 구조에 따라 유추
        # 보통 data_path/labels/category/file.txt 또는 data_path/labels/file.txt
        # 안전한 방법: 경로 치환
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        gts = []
        raw_gts = load_gt(label_path)
        for g in raw_gts:
            gts.append({'cls': int(g[0]), 'box': g[1:], 'matched': False})
            
        # 3. 매칭 (클래스별로 수행)
        # 이미지 내 통계
        img_tp = 0
        img_fp = 0
        img_fn = 0
        
        unique_classes = set([p['cls'] for p in preds] + [g['cls'] for g in gts])
        
        for cls_id in unique_classes:
            cls_preds = [p for p in preds if p['cls'] == cls_id]
            cls_gts = [g for g in gts if g['cls'] == cls_id]
            
            # 탐욕적 매칭 (Greedy Matching)
            # IoU 행렬 계산
            matches = []
            for i, p in enumerate(cls_preds):
                for j, g in enumerate(cls_gts):
                    iou = calculate_iou(p['box'], g['box'])
                    if iou >= iou_thres:
                        matches.append((iou, i, j))
            
            # IoU 높은 순으로 정렬
            matches.sort(key=lambda x: x[0], reverse=True)
            
            used_p = set()
            used_g = set()
            
            for iou, p_idx, g_idx in matches:
                if p_idx not in used_p and g_idx not in used_g:
                    used_p.add(p_idx)
                    used_g.add(g_idx)
                    
                    # 카운트 증가
                    class_stats[cls_id]['TP'] += 1
                    img_tp += 1
            
            # 남은 예측은 FP
            # 남은 GT는 FN
            fp_count = len(cls_preds) - len(used_p)
            fn_count = len(cls_gts) - len(used_g)
            
            class_stats[cls_id]['FP'] += fp_count
            class_stats[cls_id]['FN'] += fn_count
            
            img_fp += fp_count
            img_fn += fn_count

        # 이미지별 지표 계산
        # Precision = TP / (TP + FP) (예측이 없으면 1.0)
        img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 1.0
        if (img_tp + img_fp) == 0 and len(gts) > 0: # 예측도 없고 GT는 있는데? -> Precision은 정의상 분모0일때 1? 보통 예측 안했으니 틀린건 없다. 
             pass # 1.0 유지

        # Recall = TP / (TP + FN) (GT가 없으면 1.0)
        img_rec = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 1.0
        
        image_results.append({
            'image': os.path.basename(img_path),
            'precision': img_prec,
            'recall': img_rec,
            'TP': img_tp,
            'FP': img_fp,
            'FN': img_fn
        })

    # --- Global Defect Level Aggregation ---
    total_tp = sum(class_stats[c]['TP'] for c in class_stats)
    total_fp = sum(class_stats[c]['FP'] for c in class_stats)
    total_fn = sum(class_stats[c]['FN'] for c in class_stats)
    
    global_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # --- Global Image Level Aggregation (Binary OK/NG) ---
    img_tp_count = 0  # GT > 0, Pred > 0 (Correct NG)
    img_fp_count = 0  # GT = 0, Pred > 0 (Overkill)
    img_fn_count = 0  # GT > 0, Pred = 0 (Leakage)
    img_tn_count = 0  # GT = 0, Pred = 0 (Correct OK)
    
    for res in image_results:
        # Binary Classification based on counts
        is_gt_ng = (res['TP'] + res['FN']) > 0  # Any GT defect?
        is_pred_ng = (res['TP'] + res['FP']) > 0 # Any Pred defect?
        
        if is_gt_ng and is_pred_ng:
            img_tp_count += 1
        elif not is_gt_ng and is_pred_ng:
            img_fp_count += 1
        elif is_gt_ng and not is_pred_ng:
            img_fn_count += 1
        else: # not is_gt_ng and not is_pred_ng
            img_tn_count += 1
            
    total_images = len(image_results)
    total_clean_images = img_fp_count + img_tn_count
    total_defective_images = img_tp_count + img_fn_count
    
    # Image Metrics
    img_precision = img_tp_count / (img_tp_count + img_fp_count) if (img_tp_count + img_fp_count) > 0 else 0.0
    img_recall = img_tp_count / (img_tp_count + img_fn_count) if (img_tp_count + img_fn_count) > 0 else 0.0
    
    far = img_fp_count / total_clean_images if total_clean_images > 0 else 0.0  # False Alarm Rate
    leakage = img_fn_count / total_defective_images if total_defective_images > 0 else 0.0 # Leakage Rate

    # --- 최종 보고서 출력 ---
    print(f"\n{'='*20} 최종 평가 결과 {'='*20}")
    print(f"{'메트릭':<40} | {'값':<15}")
    print("-" * 60)
    print(f"{'[Standard] mAP50':<40} | {map50:.4f}")
    print(f"{'[Standard] mAP50-95':<40} | {map50_95:.4f}")
    print(f"{'[Speed] FPS':<40} | {fps:.2f}")
    print(f"{'[Speed] Latency (ms)':<40} | {avg_latency:.2f}")
    print("-" * 60)
    
    # Class-wise Standard Results
    if val_results:
        print(f"{'[Standard Class-wise Results]':<40}")
        print(f"{'Class':<20} | {'mAP50':<10} | {'mAP50-95':<10}")
        print("-" * 45)
        names = val_results.names
        for i, cls_idx in enumerate(val_results.box.ap_class_index):
            idx = int(cls_idx)
            name = names[idx]
            c_map50 = val_results.box.class_result(i)[2]
            c_map50_95 = val_results.box.class_result(i)[3]
            print(f"{name:<20} | {c_map50:.4f}     | {c_map50_95:.4f}")
        print("-" * 60)

    print(f"{'[Global Defect] Precision (Total Micro)':<40} | {global_prec:.4f}")
    print(f"{'[Global Defect] Recall (Total Micro)':<40} | {global_rec:.4f}")
    print("-" * 60)
    print(f"{'[Global Image] OK/NG Precision':<40} | {img_precision:.4f}")
    print(f"{'[Global Image] OK/NG Recall':<40} | {img_recall:.4f}")
    print(f"{'[Global Image] False Alarm Rate (과검출)':<40} | {far:.4f} ({img_fp_count}/{total_clean_images})")
    print(f"{'[Global Image] Leakage Rate (미검출)':<40} | {leakage:.4f} ({img_fn_count}/{total_defective_images})")
    print("-" * 60)
    print(f"이미지 판정 현황: TP={img_tp_count}, TN={img_tn_count}, FP(Overkill)={img_fp_count}, FN(Leakage)={img_fn_count}")
    print(f"  * 참고: 전체 이미지 {total_images}장 중 정상(Clean) {total_clean_images}장, 불량(Defective) {total_defective_images}장")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Defect Detection Custom Metrics Evaluation")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt)')
    parser.add_argument('--data', type=str, default='PCB_DATASET/data.yaml', help='Path to data.yaml')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device')
    
    args = parser.parse_args()
    
    # test_images.txt 경로 유추
    # data.yaml이 있는 폴더에 test_images.txt가 있다고 가정
    data_dir = os.path.dirname(args.data)
    test_txt = os.path.join(data_dir, 'test_images.txt')
    
    if not os.path.exists(test_txt):
        print(f"오류: 테스트 이미지 리스트를 찾을 수 없습니다: {test_txt}")
        # data.yaml 파싱해서 test 경로 찾기 시도
        with open(args.data, 'r') as f:
            d = yaml.safe_load(f)
            if 'test' in d:
                # d['test']가 txt 파일 경로일 수도 있고 폴더일 수도 있음
                # 하지만 여기서는 dataset.py가 생성한 구조를 따름
                potential_path = d['test']
                if os.path.isfile(potential_path):
                    test_txt = potential_path
    
    if not os.path.exists(test_txt):
         print("테스트 셋을 찾을 수 없어 실행을 중단합니다.")
    else:
        evaluate_metrics(args.weights, args.data, test_txt, args.imgsz, device=args.device)
