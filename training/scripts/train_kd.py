import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

# ==============================================================================
# 유틸리티: Manual Forward (수동 순전파)
# ==============================================================================



# ==============================================================================
# 1. Feature Adapter (채널 변환기)
# ==============================================================================
class FeatureAdapter(nn.Module):
    """
    Student 모델의 Feature Map 채널 수를 Teacher 모델의 채널 수와 맞추기 위한 어댑터입니다.
    1x1 Convolution을 사용하여 차원을 변경합니다.
    """

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        # 각 스케일별(P3, P4, P5) 어댑터 생성
        self.adapters = nn.ModuleList(
            [nn.Conv2d(s, t, kernel_size=1, stride=1, padding=0) for s, t in zip(student_channels, teacher_channels)]
        )

    def forward(self, features):
        # 입력된 각 Feature Map에 대해 어댑터 적용
        return [adapter(f) for adapter, f in zip(self.adapters, features)]


# ==============================================================================
# 2. Hook & Loss (지식 증류 로직)
# ==============================================================================
class KDLoss(v8DetectionLoss):
    """
    YOLOv8의 기본 Loss에 Knowledge Distillation(KD) Loss를 추가한 클래스입니다.
    - Logit KD: Box(LD), Class(KL)
    - Feature KD: MSE 또는 CWD (Channel-wise Distillation)
    """

    def __init__(self, model, teacher_model, alpha_box=0.1, alpha_cls=0.5, beta=1.0, T=4.0):
        super().__init__(model)
        self.model = model
        self.teacher_model = teacher_model

        # 하이퍼파라미터
        self.alpha_box = alpha_box  # Box Loss 가중치
        self.alpha_cls = alpha_cls  # Class Loss 가중치
        self.beta = beta  # Feature Loss 가중치
        self.T = T  # Temperature (Softmax 완화 계수)
        self.feature_loss_type = "mse"  # 기본값 (mse, cwd, at, fgfi)

        # 데이터 저장소
        self.student_features = {}
        self.teacher_features = {}
        self.student_logits = {"box": {}, "cls": {}}
        self.teacher_logits = {"box": {}, "cls": {}}

        # Hook 관리 핸들
        self.hook_handles = []

        # 타겟 레이어 인덱스 (Trainer에서 주입됨)
        self.feature_layers = []
        self.teacher_feature_layers = []

        # Teacher 모델 초기화
        self._init_teacher()

        # Hook 등록은 Trainer에서 레이어 주입 후 명시적으로 호출합니다.
        # self.restore_hooks()

    def _init_teacher(self):
        """Teacher 모델을 평가 모드로 고정합니다."""
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        # BatchNorm 통계 업데이트 방지
        for m in self.teacher_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.eval()

    def remove_hooks(self):
        """등록된 모든 Hook을 제거합니다."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def restore_hooks(self):
        """저장된 레이어 정보를 바탕으로 Hook을 다시 등록합니다."""
        self.remove_hooks()

        # 1. Feature Hooks (Beta > 0 인 경우에만)
        if self.beta > 0:
            if not self.feature_layers or not self.teacher_feature_layers:
                 print("[KDLoss] 경고: Feature Layer 인덱스가 비어 있습니다. Feature KD가 동작하지 않을 수 있습니다.")
            
            self._register_hooks(self.model, self.student_features, "Student", self.feature_layers)
            self._register_hooks(self.teacher_model, self.teacher_features, "Teacher", self.teacher_feature_layers)

        # 2. Logit Hooks (Alpha > 0 인 경우에만)
        if self.alpha_box > 0 or self.alpha_cls > 0:
            self._register_head_hooks(self.model, self.student_logits, "Student")
            self._register_head_hooks(self.teacher_model, self.teacher_logits, "Teacher")

        # Teacher 모델 다시 한 번 Eval 고정 (안전장치)
        self._init_teacher()

    def _register_hooks(self, model, storage, prefix, layers):
        """특정 레이어에 Forward Hook을 등록합니다."""
        if not layers:
            print(f"[{prefix}] 경고: Hook을 등록할 레이어 리스트가 비어 있습니다.")
            return

        print(f"[{prefix}] Hook 등록 레이어: {layers}")

        # 모델 내부 모듈 리스트 접근
        inner = model.model if hasattr(model, "model") else model
        module_list = inner.model if hasattr(inner, "model") else inner

        for i in layers:
            if i < len(module_list):
                # Use picklable FeatureHook class
                h = module_list[i].register_forward_hook(FeatureHook(storage, i))
                self.hook_handles.append(h)

    def _register_head_hooks(self, model, storage, prefix):
        """Detect Head(최종 출력단)에 Hook을 등록합니다."""
        head = None

        # Detect Head 찾기 탐색 로직
        inner = model.model if hasattr(model, "model") else model
        module_list = inner.model if hasattr(inner, "model") else inner

        if len(module_list) > 0:
            last = module_list[-1]
            if "Detect" in str(type(last)):
                head = last

        if head is None:
            print(f"[{prefix}] 경고: Detect Head를 찾을 수 없습니다. (Logit KD 불가)")
            return

        print(f"[{prefix}] Head 감지됨: {type(head).__name__}. Output Hook 등록 중...")

        # Box Head (cv2)
        if hasattr(head, "cv2"):
            for i, m in enumerate(head.cv2):
                self.hook_handles.append(m.register_forward_hook(FeatureHook(storage["box"], i)))

        # Class Head (cv3)
        if hasattr(head, "cv3"):
            for i, m in enumerate(head.cv3):
                self.hook_handles.append(m.register_forward_hook(FeatureHook(storage["cls"], i)))

    def _at_loss(self, s_feat, t_feat):
        """Attention Transfer Loss (Zagoruyko et al., 2017)"""
        # [B, C, H, W] -> [B, H, W] (Sum of absolute values across channels)
        s_att = torch.sum(torch.abs(s_feat), dim=1)
        t_att = torch.sum(torch.abs(t_feat), dim=1)

        # Normalize: L2-norm spatial map
        # Flatten [B, H, W] -> [B, H*W] for normalization
        s_att = F.normalize(s_att.view(s_att.size(0), -1), p=2, dim=1)
        t_att = F.normalize(t_att.view(t_att.size(0), -1), p=2, dim=1)

        # MSE between normalized attention maps
        # [Fix] Scale invariance: Sum per image (Max range 0~4) to match Task Loss magnitude
        return F.mse_loss(s_att, t_att, reduction="sum") / s_att.size(0)

    def _fgfi_loss(self, s_feat, t_feat, batch):
        """
        Fine-Grained Feature Imitation (FGFI) Loss (Vectorized)
        논문: Distilling Object Detectors with Fine-grained Feature Imitation (CVPR 2019)
        """
        # 1. Basic MSE (Pixel-wise)
        diff = (s_feat - t_feat) ** 2

        # 2. Mask 생성 (B, 1, H, W)
        B, C, H, W = s_feat.shape
        mask = torch.zeros((B, 1, H, W), device=s_feat.device)

        if batch is not None and "bboxes" in batch and "batch_idx" in batch:
            bboxes = batch["bboxes"]  # (M, 4) Normalized xywh
            batch_idx = batch["batch_idx"]  # (M) Image Index

            # Vectorized Mask Generation is tricky with variable number of boxes per image.
            # Using optimized loop with broadcasting.
            
            # Normalized -> Absolute Coordinates
            # bboxes: [cx, cy, w, h]
            x1 = (bboxes[:, 0] - bboxes[:, 2] / 2) * W
            y1 = (bboxes[:, 1] - bboxes[:, 3] / 2) * H
            x2 = (bboxes[:, 0] + bboxes[:, 2] / 2) * W
            y2 = (bboxes[:, 1] + bboxes[:, 3] / 2) * H
            
            x1 = x1.clamp(0, W).long()
            y1 = y1.clamp(0, H).long()
            x2 = x2.clamp(0, W).long()
            y2 = y2.clamp(0, H).long() # x2, y2 exclusive for slicing? slicing is exclusive.
            
            # Since generating masks purely via vectorization for all images efficiently is complex,
            # we optimize the inner loop.
            for i in range(B):
                # Indices for current image
                idx = (batch_idx == i)
                if not idx.any():
                    continue
                    
                b_x1, b_y1, b_x2, b_y2 = x1[idx], y1[idx], x2[idx], y2[idx]
                
                # Fill mask for each box
                # Still loop over boxes needed unless we use a grid method, but loop over boxes is usually fast enough if count is low.
                # Optimized:
                for j in range(len(b_x1)):
                    mask[i, :, b_y1[j]:b_y2[j], b_x1[j]:b_x2[j]] = 1.0

        # 3. Masked Mean Calculation
        intersection = (diff * mask).sum()
        mask_sum = mask.sum()

        if mask_sum > 0:
            # [Fix] Scale-wise Normalization
            loss = (intersection / mask_sum) / C
        else:
            loss = torch.tensor(0.0, device=s_feat.device)

        return loss

    def _cwd_loss(self, y_s, y_t):
        """Channel-wise Distillation Loss (KL Divergence based)"""
        # [Fix] Teacher와 Student의 채널 수나 Spatial 차원이 다를 수 있으므로
        # 각각의 shape를 기반으로 flatten 해야 안전합니다.
        B_s, C_s, H_s, W_s = y_s.shape
        B_t, C_t, H_t, W_t = y_t.shape

        # 채널별 공간 분포(H*W)를 확률 분포로 변환
        # [B, C, H*W]
        y_s_flat = y_s.view(B_s, C_s, -1)
        y_t_flat = y_t.view(B_t, C_t, -1)

        s_prob = F.log_softmax(y_s_flat / self.T, dim=-1)  # Log Softmax for Student
        t_prob = F.softmax(y_t_flat / self.T, dim=-1)  # Softmax for Teacher

        # KL Divergence 계산
        # [Fix] Channel Normalization: 채널 수로 나누어 Feature Loss 스케일을 조정합니다.
        loss = (F.kl_div(s_prob, t_prob, reduction="batchmean") * (self.T**2)) / C_s
        return loss

    def __call__(self, preds, batch):
        # 1. 기본 Task Loss 계산 (Box, Cls, DFL)
        loss, loss_items = super().__call__(preds, batch)

        if hasattr(loss, "numel") and loss.numel() > 1:
            loss = loss.sum()

        kd_loss_box = torch.tensor(0.0, device=self.device)
        kd_loss_cls = torch.tensor(0.0, device=self.device)
        kd_loss_feature = torch.tensor(0.0, device=self.device)

        # 1. Teacher Forward (No Gradients)
        with torch.no_grad():
            img = batch["img"]

            # Teacher 입력 타입 맞추기 (Safe Casting)
            t_param = next(self.teacher_model.parameters())
            # Ensure device and dtype match teacher
            if img.device != t_param.device or img.dtype != t_param.dtype:
                img = img.to(device=t_param.device, dtype=t_param.dtype, non_blocking=True)

            # [Refinement] Resolution Matching (Input Synchronization)
            # Teacher에게 Student와 동일한 해상도의 이미지를 입력합니다.
            
            # Standard Forward (Hooks will capture features/logits)
            self.teacher_model(img)

        # 2. Student Logit 수집
        # (YOLO Logit은 Head에서 계산되므로, 이미 Hooks에 의해 student_logits에 담겨 있음)

        # 3. Logit KD Loss 계산
        if self.alpha_box > 0 or self.alpha_cls > 0:
            # Box Logit Loss (Only if alpha_box > 0)
            if self.alpha_box > 0:
                for i, s_box in self.student_logits["box"].items():
                    if i in self.teacher_logits["box"]:
                        t_box = self.teacher_logits["box"][i]
                        # Shape: [B, 4*RegMax, H, W]
                        B, C, H, W = s_box.shape

                        # 해상도 불일치 시 보간 (Teacher -> Student 크기로)
                        if t_box.shape[-2:] != (H, W):
                            t_box = F.interpolate(t_box, size=(H, W), mode="bilinear", align_corners=False)

                        reg_max = C // 4
                        s_dist = s_box.view(B, 4, reg_max, H, W)
                        t_dist = t_box.view(B, 4, reg_max, H, W)

                        loss_box = F.kl_div(
                            F.log_softmax(s_dist / self.T, dim=2),
                            F.softmax(t_dist / self.T, dim=2),
                            reduction="batchmean",
                        )
                        # [Fix] Spatial Normalization: Grid 칸 수만큼 나누어 손실 스케일 조정함
                        kd_loss_box += (loss_box * (self.T**2)) / (H * W)

            # Class Logit Loss (Only if alpha_cls > 0)
            if self.alpha_cls > 0:
                for i, s_cls in self.student_logits["cls"].items():
                    if i in self.teacher_logits["cls"]:
                        t_cls = self.teacher_logits["cls"][i]

                        # 해상도 불일치 시 보간
                        if t_cls.shape[-2:] != s_cls.shape[-2:]:
                            t_cls = F.interpolate(t_cls, size=s_cls.shape[-2:], mode="bilinear", align_corners=False)

                        # 클래스 수 불일치 처리 (S:6 vs T:80 등)
                        if s_cls.shape[1] != t_cls.shape[1]:
                            t_cls = t_cls[:, : s_cls.shape[1], ...]

                        loss_cls = F.kl_div(
                            F.log_softmax(s_cls / self.T, dim=1),
                            F.softmax(t_cls / self.T, dim=1),
                            reduction="batchmean",
                        )
                        # [Fix] Spatial Normalization: Grid 칸 수만큼 나누어 손실 스케일 조정
                        kd_loss_cls += (loss_cls * (self.T**2)) / (s_cls.shape[-1] * s_cls.shape[-2])

        # 4. Feature KD Loss 계산
        if self.beta > 0 and hasattr(self.model, "kd_adapters"):
            # 수집된 Feature 가져오기
            s_feats = [self.student_features[i] for i in self.feature_layers if i in self.student_features]
            t_feats = [self.teacher_features[i] for i in self.teacher_feature_layers if i in self.teacher_features]

            if len(s_feats) > 0 and len(s_feats) == len(t_feats):

                # [Fix] Compatibility: 일부 모델(v8n 등)은 Feature를 dict나 list로 반환할 수 있음
                def extract_tensor(f):
                    if isinstance(f, torch.Tensor):
                        return f
                    if isinstance(f, dict):
                        return list(f.values())[0]
                    if isinstance(f, (list, tuple)):
                        return f[0]
                    return f

                # [Fix] AT(Attention Transfer)는 Adapter 없이 직접 비교하는 것이 더 효과적입니다.
                if self.feature_loss_type == "at":
                    for s_f, t_f in zip(s_feats, t_feats):
                        s_f = extract_tensor(s_f)
                        t_f = extract_tensor(t_f)

                        if s_f.shape[-2:] != t_f.shape[-2:]:
                            s_f = F.interpolate(s_f, size=t_f.shape[-2:], mode="bilinear", align_corners=False)

                        kd_loss_feature += self._at_loss(s_f, t_f)

                # [New] FGFI (Masked Feature KD)
                elif self.feature_loss_type == "fgfi":
                    # Adapter 적용 (Channel Matching)
                    s_feats = [extract_tensor(f) for f in s_feats]
                    t_feats = [extract_tensor(f) for f in t_feats]

                    if hasattr(self.model, "kd_adapters"):
                        s_feats_adapted = self.model.kd_adapters(s_feats)
                    else:
                        s_feats_adapted = s_feats

                    fgfi_losses = []
                    for s_f, t_f in zip(s_feats_adapted, t_feats):
                        # [Fix] Resolution Matching: Teacher Input Synchronization으로 인해
                        # 이제 Feature Map 크기가 자연스럽게 일치해야 함.
                        # 만약 불일치한다면(Stride 차이 등), 그때만 보간.
                        if s_f.shape[-2:] != t_f.shape[-2:]:
                            s_f = F.interpolate(s_f, size=t_f.shape[-2:], mode="bilinear", align_corners=False)

                        fgfi_losses.append(self._fgfi_loss(s_f, t_f, batch))

                    # [Fix] Scale-wise Normalization: Sum 대신 Mean 사용
                    if len(fgfi_losses) > 0:
                        kd_loss_feature += sum(fgfi_losses) / len(fgfi_losses)

                else:
                    # CWD, MSE는 채널 수가 같아야 하므로 Adapter 필수
                    s_feats = [extract_tensor(f) for f in s_feats]
                    t_feats = [extract_tensor(f) for f in t_feats]

                    s_feats_adapted = self.model.kd_adapters(s_feats)

                    for s_f, t_f in zip(s_feats_adapted, t_feats):
                        if s_f.shape[-2:] != t_f.shape[-2:]:
                            s_f = F.interpolate(s_f, size=t_f.shape[-2:], mode="bilinear", align_corners=False)

                        if self.feature_loss_type == "cwd":
                            kd_loss_feature += self._cwd_loss(s_f, t_f)
                        else:
                            kd_loss_feature += F.mse_loss(s_f, t_f)

            # 다음 배치를 위해 초기화
            self.student_features.clear()
            self.teacher_features.clear()

        # Logit 저장소 초기화
        self.student_logits["box"].clear()
        self.student_logits["cls"].clear()
        self.teacher_logits["box"].clear()
        self.teacher_logits["cls"].clear()

        # 5. 최종 Loss 합산
        total_loss = loss + (self.alpha_box * kd_loss_box) + (self.alpha_cls * kd_loss_cls) + (self.beta * kd_loss_feature)

        # 6. 로깅용 값 저장
        self.loss_diagnostics = {
            "task_loss": loss.detach(),
            "kd_box": kd_loss_box.detach(),
            "kd_cls": kd_loss_cls.detach(),
            "kd_feature": kd_loss_feature.detach(),
        }

        return total_loss, loss_items


# ==============================================================================
# Helper Classes (Pickling Safe)
# ==============================================================================
class FeatureHook:
    """
    Hook to capture feature maps or logits.
    Must be a global class to be picklable by torch.save().
    """
    def __init__(self, storage, idx):
        self.storage = storage
        self.idx = idx

    def __call__(self, module, input, output):
        self.storage[self.idx] = output


# ==============================================================================
# 3. KD Logger (커스텀 로깅)
# ==============================================================================
class KDLogger:
    """
    KD 학습 과정을 더 친절하게 출력하기 위한 로거 클래스입니다.
    Ultralytics Trainer의 Callback 시스템을 활용합니다.
    """

    def __init__(self):
        self.loss_sum = {"box": 0.0, "cls": 0.0, "feat": 0.0}
        self.count = 0
        self.best_fitness = -1.0

    def on_train_epoch_start(self, trainer):
        self.loss_sum = {"box": 0.0, "cls": 0.0, "feat": 0.0}
        self.count = 0

    def on_train_batch_end(self, trainer):
        # KDLoss의 loss_diagnostics에 접근하여 값 누적
        crit = getattr(trainer.model, "criterion", None)
        if crit and hasattr(crit, "loss_diagnostics"):
            d = crit.loss_diagnostics
            # Tensor -> float 변환
            self.loss_sum["box"] += d.get("kd_box", torch.tensor(0.0)).item()
            self.loss_sum["cls"] += d.get("kd_cls", torch.tensor(0.0)).item()
            self.loss_sum["feat"] += d.get("kd_feature", torch.tensor(0.0)).item()
            self.count += 1

    def on_train_epoch_end(self, trainer):
        cnt = max(1, self.count)
        box_avg = self.loss_sum["box"] / cnt
        cls_avg = self.loss_sum["cls"] / cnt
        feat_avg = self.loss_sum["feat"] / cnt

        print("\n────────────────────────────────────────────────────────────")
        print(f"  🎓 KD Loss Summary (Epoch {trainer.epoch + 1})")
        print(f"     - Box KD  : {box_avg:.4f} (위치 따라하기)")
        print(f"     - Cls KD  : {cls_avg:.4f} (분류 따라하기)")
        print(f"     - Feat KD : {feat_avg:.4f} (특징 따라하기)")
        print("────────────────────────────────────────────────────────────")

    def on_fit_epoch_end(self, trainer):
        # Best Model 갱신 여부 확인
        # trainer.fitness는 Tensor일수도, float일수도 있음
        if isinstance(trainer.fitness, torch.Tensor):
            curr_fit = trainer.fitness.item()
        else:
            curr_fit = float(trainer.fitness)

        if curr_fit > self.best_fitness:
            self.best_fitness = curr_fit
            print("  🚀 신기록 달성! (New Best Model Saved)")
            print(f"     - Fitness : {curr_fit:.4f}")
            if hasattr(trainer.validator, "metrics"):
                m = trainer.validator.metrics
                print(f"     - mAP50   : {m.results_dict.get('metrics/mAP50(B)', 0):.4f}")
                print(f"     - mAP50-95: {m.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
            print("────────────────────────────────────────────────────────────\n")


# ==============================================================================
# 4. KD Trainer (학습기)
# ==============================================================================
class KnowledgeDistillationTrainer(DetectionTrainer):
    """
    YOLO의 DetectionTrainer를 확장하여 KD 기능을 통합한 클래스입니다.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """모델을 로드하고 KD 관련 모듈(Teacher, Adapter, Loss)을 설정합니다."""
        # 1. Student 모델 로드 (부모 클래스 메서드 사용)
        model = super().get_model(cfg, weights, verbose)

        # 2. Teacher 모델 로드
        kd_args = getattr(self, "kd_args", {})
        t_path = kd_args.get("teacher_model")
        if not t_path:
            raise ValueError("KD Error: Teacher 모델 경로가 없습니다.")

        print(f"[KD] Teacher 모델 로드 중: {t_path}")
        teacher = YOLO(t_path)
        # [Fix] Teacher의 내부 모델을 명시적으로 device로 이동
        if hasattr(teacher, "model"):
             teacher.model.to(self.device)
        else:
             teacher.to(self.device)
        
        model.to(self.device)

        # [KD Auto-Detect] Teacher의 학습 해상도 감지
        try:
            if hasattr(teacher, "ckpt") and "train_args" in teacher.ckpt:
                t_imgsz = teacher.ckpt["train_args"]["imgsz"]
                print(f"[KD] Teacher 학습 해상도 감지됨: {t_imgsz}")
            else:
                # ckpt가 없는 경우 args 확인
                t_imgsz = getattr(teacher.model, "args", {}).get("imgsz", 1280)
                print(f"[KD] Teacher 해상도 추정 (Fallback): {t_imgsz}")
        except Exception as e:
            print(f"[KD Warning] Teacher 해상도 감지 실패 (Default 1280): {e}")
            t_imgsz = 1280

        self.teacher_imgsz = t_imgsz

        self.teacher_imgsz = t_imgsz

        # 3. Feature Layer 자동 감지 (Dynamic Layer Detection via Detect Head)
        def find_layers_via_head(m, name):
            inner = m.model if hasattr(m, "model") else m
            layers = inner.model if hasattr(inner, "model") else inner

            det_idx = -1
            det_layer = None
            for i, layer in enumerate(layers):
                if "Detect" in str(type(layer)):
                    det_idx = i
                    det_layer = layer
                    break

            if det_idx == -1:
                print(f"[KD Error] {name}에서 Detect Layer를 찾을 수 없습니다. 기본값(v11)을 사용합니다.")
                return [16, 19, 22]

            # Detect Layer의 .f 속성을 통해 입력 레이어 추적 (정확함)
            if hasattr(det_layer, 'f'):
                found = det_layer.f
                if isinstance(found, int): found = [found] # 단일 입력인 경우 리스트로 변환
                print(f"[KD] {name} 레이어 감지 (via .f): Detect({det_idx}) Inputs -> {found}")
                return found
            
            # Fallback: 역추적 (P3, P4, P5)
            p5 = det_idx - 1
            p4 = det_idx - 4
            p3 = det_idx - 7
            print(f"[KD Warning] {name} Detect.f 속성 없음. 위치 추정: {p3}, {p4}, {p5}")
            return [p3, p4, p5]

        self.verified_s_layers = find_layers_via_head(model, "Student")
        self.verified_t_layers = find_layers_via_head(teacher, "Teacher")

        # 채널 수 자동 감지 (Robust Hook-based Method)
        def get_channels_robust(m, layers, device, name="Model"):
            # print(f"[KD] {name} 레이어 채널 감지 중 (Device: {device})...")
            inner = m.model if hasattr(m, "model") else m
            module_list = inner.model if hasattr(inner, "model") else inner
            
            channels = {}
            handles = []
            
            for i in layers:
                if i < len(module_list):
                    # Use picklable FeatureHook for robust detection as well (consistency)
                    handles.append(module_list[i].register_forward_hook(FeatureHook(channels, i)))
            
            # [Fix] 64x64 led to issues, using standard 640.
            # Using CPU for safe detection to avoid GPU-specific layer errors (e.g. Concat).
            sz = 640 
            dummy_shape = (1, 3, sz, sz) 
            
            # [Fix] Run forward on the outer model (m) to ensure skip connections work,
            # but keep hooks on the inner module_list to capture layer outputs.
            # Using CPU for safe detection.
            p = next(inner.parameters())
            original_device = p.device
            target_device = torch.device('cpu')
            
            # We need to move the WHOLE model, not just inner
            m.to(target_device)
            
            # Save training state and set to eval
            was_training = m.training
            m.eval()
            
            try:
                dummy_img = torch.zeros(dummy_shape, device=target_device)
                with torch.no_grad():
                    # Execute OUTER model (m) which handles the forward logic (Sequential + Skip)
                    m(dummy_img) 
            except Exception as e:
                print(f"[KD Warning] 채널 감지 Forward 실패 ({name}): {e}. 기본값(256) 사용.")
            finally:
                # Restore
                if hasattr(m, 'model') and isinstance(m.model, torch.nn.Module):
                    m.model.train(was_training)
                    if hasattr(m.model, 'to'): m.model.to(original_device)
                elif isinstance(m, torch.nn.Module):
                    m.train(was_training)
                    m.to(original_device)
            
            # Clean up
            for h in handles: h.remove()
            
            result = []
            for i in layers:
                if i in channels:
                    # FeatureHook stores the full output tensor
                    feat = channels[i]
                    if isinstance(feat, torch.Tensor):
                        result.append(feat.shape[1])
                    else:
                        result.append(256)
                else:
                    result.append(256)
            print(f"  -> 감지된 채널: {result}")
            return result

        # 여기서는 Student/Teacher 모델 객체에서 직접 채널 정보를 읽음
        s_channels = get_channels_robust(model, self.verified_s_layers, self.device, name="Student")
        t_channels = get_channels_robust(teacher, self.verified_t_layers, self.device, name="Teacher")

        print(f"[KD] Verified Student Layers: {self.verified_s_layers}")
        print(f"[KD] Verified Teacher Layers: {self.verified_t_layers}")
        print(f"[KD] Detected Student Channels: {s_channels}")
        print(f"[KD] Detected Teacher Channels: {t_channels}")

        # 4. Adapter 초기화 및 부착
        model.kd_adapters = FeatureAdapter(s_channels, t_channels).to(self.device)
        
        # Adapter 파라미터 학습 가능 설정
        for p in model.kd_adapters.parameters():
            p.requires_grad = True

        # [Fix] KDLoss 부모 클래스(v8DetectionLoss)가 model.args를 참조하므로
        # Loss 초기화 전에 args를 먼저 모델에 주입해야 합니다.
        if not hasattr(model, "args"):
            model.args = self.args

        # 5. KD Loss 교체
        model.criterion = KDLoss(
            model,
            teacher.model,
            alpha_box=kd_args.get("kd_alpha_box", 0.1),
            alpha_cls=kd_args.get("kd_alpha_cls", 0.5),
            beta=kd_args.get("kd_beta", 1.0),
            T=kd_args.get("kd_T", 4.0),
        )
        model.criterion.feature_loss_type = kd_args.get("kd_feature_type", "mse")
        model.criterion.feature_layers = self.verified_s_layers
        model.criterion.teacher_feature_layers = self.verified_t_layers

        # 저장된 Args 주입 (Loss 내부 사용용)
        if not hasattr(model, "args"):
            model.args = self.args

        # Hook 활성화
        model.criterion.restore_hooks()

        return model


# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Knowledge Distillation (Config based)")
    parser.add_argument("--config", type=str, default="configs/config_kd.yaml", help="설정 파일 경로 (.yaml)")
    args = parser.parse_args()

    # 1. 설정 로드
    if not os.path.exists(args.config):
        print(f"[Error] 설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 필수 값 체크 및 절대 경로 변환
    if not config.get("teacher_model"):
        print("[Error] Config에 'teacher_model' 경로가 없습니다.")
        sys.exit(1)
    if not config.get("student_model"):
        print("[Error] Config에 'student_model' 경로가 없습니다.")
        sys.exit(1)

    print(f"\n[KD] 학습 시작 (Config: {args.config})")
    print(f" - Teacher: {config['teacher_model']}")
    print(f" - Student: {config['student_model']}")

    # 3. Trainer 인자 준비 (YOLO 호환성 유지)
    # config의 모든 키를 overrides로 사용하되, KD 전용 키는 분리
    kd_keys = [
        "teacher_model",
        "student_model",
        "kd_alpha_box",
        "kd_alpha_cls",
        "kd_beta",
        "kd_T",
        "kd_feature_type",
    ]

    yolo_overrides = config.copy()
    yolo_overrides["model"] = config["student_model"]  # Student 모델을 기준 모델로 설정
    # yolo_overrides["save_dir"] = config.get("project", "runs/kd_feature") # save_dir 설정 제거 (Ultralytics Convention 따름)
    
    # [Fix] Project/Name ensure - Save in training/runs/kd explicitly to avoid nesting
    script_dir = Path(__file__).resolve().parent.parent
    base_runs_dir = script_dir / "runs"
    
    # 설정파일에서 project 읽기 (기본값 'kd')
    project_name = config.get("project", "kd")
    if project_name == "runs": project_name = "kd"
    elif project_name.startswith("runs/"): project_name = project_name.replace("runs/", "", 1)
    
    yolo_overrides["project"] = str(base_runs_dir / project_name)
        
    if "name" not in yolo_overrides or yolo_overrides["name"] is None:
        # 실험명 자동 생성 또는 설정값 사용
        s_stem = Path(config["student_model"]).stem
        t_stem = Path(config["teacher_model"]).stem
        yolo_overrides["name"] = config.get("exp_name", f"KD_{s_stem}_from_{t_stem}")
        
    # [Fix] Optimizer Case Sensitivity (Adamw -> AdamW)
    opt = yolo_overrides.get("optimizer", "auto")
    if isinstance(opt, str) and opt.lower() == "adamw":
        yolo_overrides["optimizer"] = "AdamW"

    # KD 전용 인자는 yolo_overrides에서 제거 (Ultralytics Warning 방지)
    kd_config = {}
    for k in kd_keys:
        if k in yolo_overrides:
            kd_config[k] = yolo_overrides.pop(k) # 추출하여 kd_config로 이동

    # 4. Trainer 실행
    trainer = KnowledgeDistillationTrainer(overrides=yolo_overrides)
    trainer.kd_args = kd_config
    
    # Logger 등록
    kd_logger = KDLogger()
    trainer.add_callback("on_train_epoch_start", kd_logger.on_train_epoch_start)
    trainer.add_callback("on_train_batch_end", kd_logger.on_train_batch_end)
    trainer.add_callback("on_train_epoch_end", kd_logger.on_train_epoch_end)
    trainer.add_callback("on_fit_epoch_end", kd_logger.on_fit_epoch_end)

    trainer.train()
