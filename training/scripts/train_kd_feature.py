import argparse
import os
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
def manual_forward(model, x):
    """
    YOLO 모델의 Forward Pass를 수동으로 실행합니다.
    
    이유:
    1. Ultralytics의 모델 래퍼(DetectionModel, Sequential 등)가 복잡하게 얽혀 있어,
       특정 환경에서 `model(x)` 호출 시 `Sequential.forward`가 실행되어 Skip Connection이 끊기는 문제 발생.
    2. 이를 방지하기 위해 각 레이어를 직접 순회하며, 필요한 Feature Map을 저장하고 연결(Concat)합니다.
    
    Args:
        model: Teacher Model 객체 (YOLO)
        x: 입력 이미지 텐서
    
    Returns:
        최종 출력 (Prediction)
    """
    y = []  # 레이어 출력 저장소 (Skip Connection 용)
    
    # 1. 내부 모델 추출 (Wrapper 벗기기)
    inner_model = model.model if hasattr(model, 'model') else model
    # 한 번 더 벗겨야 하는 경우 (DetectionModel -> Sequential)
    layers = inner_model.model if hasattr(inner_model, 'model') else inner_model
    
    # 2. 레이어 순회
    for i, m in enumerate(layers):
        if m.f != -1:  # 이전 레이어 참조가 있는 경우 (Skip Connection)
            if isinstance(m.f, int):
                # 단일 이전 레이어 참조
                x = y[m.f]
            else:
                # 다중 이전 레이어 참조 (Concat 등)
                # 주의: 저장되지 않은(None) 레이어를 참조하려 하면 에러가 발생하므로,
                # 모든 레이어를 저장하도록 로직이 개선되었습니다.
                x = [x if j == -1 else y[j] for j in m.f]
        
        # 레이어 실행
        x = m(x)
        
        # 3. 출력 저장
        # 메모리를 조금 더 쓰더라도 안정성을 위해 모든 레이어의 출력을 저장합니다.
        # YOLO 모델은 깊이가 깊지 않아(약 20~300층) 참조만 저장하는 것은 부담이 적습니다.
        y.append(x)
        
    return x

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
        self.adapters = nn.ModuleList([
            nn.Conv2d(s, t, kernel_size=1, stride=1, padding=0)
            for s, t in zip(student_channels, teacher_channels)
        ])
    
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
        self.beta = beta            # Feature Loss 가중치
        self.T = T                  # Temperature (Softmax 완화 계수)
        self.feature_loss_type = 'mse' # 기본값 (mse, cwd)
        
        # 데이터 저장소
        self.student_features = {}
        self.teacher_features = {}
        self.student_logits = {'box': {}, 'cls': {}}
        self.teacher_logits = {'box': {}, 'cls': {}}
        
        # Hook 관리 핸들
        self.hook_handles = []
        
        # 타겟 레이어 인덱스 (Trainer에서 주입됨)
        self.feature_layers = [] 
        self.teacher_feature_layers = []
        
        # Teacher 모델 초기화
        self._init_teacher()

        # Hook 등록
        self.restore_hooks()

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
            self._register_hooks(self.model, self.student_features, 'Student', self.feature_layers)
            self._register_hooks(self.teacher_model, self.teacher_features, 'Teacher', self.teacher_feature_layers)
        
        # 2. Logit Hooks (Alpha > 0 인 경우에만)
        if self.alpha_box > 0 or self.alpha_cls > 0:
            self._register_head_hooks(self.model, self.student_logits, 'Student')
            self._register_head_hooks(self.teacher_model, self.teacher_logits, 'Teacher')
            
        # Teacher 모델 다시 한 번 Eval 고정 (안전장치)
        self._init_teacher()

    def _register_hooks(self, model, storage, prefix, layers):
        """특정 레이어에 Forward Hook을 등록합니다."""
        if not layers:
            print(f"[{prefix}] 경고: Hook을 등록할 레이어 리스트가 비어 있습니다.")
            return

        print(f"[{prefix}] Hook 등록 레이어: {layers}")
        
        # 모델 내부 모듈 리스트 접근
        inner = model.model if hasattr(model, 'model') else model
        module_list = inner.model if hasattr(inner, 'model') else inner
        
        # Closure를 이용한 Hook 함수 생성
        def get_hook(storage_dict, idx):
            def hook(module, input, output):
                storage_dict[idx] = output
            return hook

        for i in layers:
            if i < len(module_list):
                h = module_list[i].register_forward_hook(get_hook(storage, i))
                self.hook_handles.append(h)
    
    def _register_head_hooks(self, model, storage, prefix):
        """Detect Head(최종 출력단)에 Hook을 등록합니다."""
        head = None
        
        # Detect Head 찾기 탐색 로직
        inner = model.model if hasattr(model, 'model') else model
        module_list = inner.model if hasattr(inner, 'model') else inner

        if len(module_list) > 0:
            last = module_list[-1]
            if 'Detect' in str(type(last)):
                head = last
        
        if head is None:
            print(f"[{prefix}] 경고: Detect Head를 찾을 수 없습니다. (Logit KD 불가)")
            return

        print(f"[{prefix}] Head 감지됨: {type(head).__name__}. Output Hook 등록 중...")

        def get_hook(storage_dict, idx):
            def hook(module, input, output):
                storage_dict[idx] = output
            return hook

        # Box Head (cv2)
        if hasattr(head, 'cv2'):
            for i, m in enumerate(head.cv2):
                self.hook_handles.append(m.register_forward_hook(get_hook(storage['box'], i)))
        
        # Class Head (cv3)
        if hasattr(head, 'cv3'):
            for i, m in enumerate(head.cv3):
                self.hook_handles.append(m.register_forward_hook(get_hook(storage['cls'], i)))

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
        
        s_prob = F.log_softmax(y_s_flat / self.T, dim=-1) # Log Softmax for Student
        t_prob = F.softmax(y_t_flat / self.T, dim=-1)     # Softmax for Teacher
        
        # KL Divergence 계산
        loss = F.kl_div(s_prob, t_prob, reduction='batchmean') * (self.T ** 2)
        return loss

    def __call__(self, preds, batch):
        # 1. 기본 Task Loss 계산 (Box, Cls, DFL)
        loss, loss_items = super().__call__(preds, batch)
        
        if hasattr(loss, 'numel') and loss.numel() > 1:
            loss = loss.sum()
        
        kd_loss_box = torch.tensor(0., device=self.device)
        kd_loss_cls = torch.tensor(0., device=self.device)
        kd_loss_feature = torch.tensor(0., device=self.device)
        
        # 2. Teacher Forward Pass
        try:
            with torch.no_grad():
                img = batch['img']
                
                # Teacher 입력 타입 맞추기
                t_param = next(self.teacher_model.parameters())
                if img.dtype != t_param.dtype:
                    img = img.to(t_param.dtype)
                
                # [KD Optimization] Cross-Resolution Distillation
                # Teacher의 학습 해상도로 리사이징하여 성능 극대화 (Auto-Detect)
                # t_imgsz가 없으면 기본값 1280 (XLarge) 또는 640을 사용
                target_sz = getattr(self, 'teacher_imgsz', 1280) 
                
                if img.shape[-1] != target_sz:
                    img = F.interpolate(img, size=(target_sz, target_sz), mode='bilinear', align_corners=False)
                
                # [디버깅] 입력 크기 확인 (Cross-Resolution 확인용)
                # print(f"[KD Debug] Teacher Input Shape: {img.shape}")

                self._init_teacher()
                
                # [중요] Manual Forward 사용 (Sequential 에러 방지)
                manual_forward(self.teacher_model, img)
                
        except Exception as e:
            print(f"[KD Error] Teacher Forward 실패: {e}")
            raise e

        # 3. Logit KD Loss 계산
        if self.alpha_box > 0 or self.alpha_cls > 0:
            # Box Logit Loss
            for i, s_box in self.student_logits['box'].items():
                if i in self.teacher_logits['box']:
                    t_box = self.teacher_logits['box'][i]
                    # Shape: [B, 4*RegMax, H, W]
                    B, C, H, W = s_box.shape
                    reg_max = C // 4
                    
                    s_dist = s_box.view(B, 4, reg_max, H, W)
                    t_dist = t_box.view(B, 4, reg_max, H, W)
                    
                    loss_box = F.kl_div(
                        F.log_softmax(s_dist / self.T, dim=2),
                        F.softmax(t_dist / self.T, dim=2),
                        reduction='batchmean'
                    )
                    kd_loss_box += loss_box * (self.T ** 2)

            # Class Logit Loss
            for i, s_cls in self.student_logits['cls'].items():
                if i in self.teacher_logits['cls']:
                    t_cls = self.teacher_logits['cls'][i]
                    
                    # 클래스 수 불일치 처리 (S:6 vs T:80 등)
                    if s_cls.shape[1] != t_cls.shape[1]:
                        t_cls = t_cls[:, :s_cls.shape[1], ...]
                    
                    loss_cls = F.kl_div(
                        F.log_softmax(s_cls / self.T, dim=1),
                        F.softmax(t_cls / self.T, dim=1),
                        reduction='batchmean'
                    )
                    kd_loss_cls += loss_cls * (self.T ** 2)

        # 4. Feature KD Loss 계산
        if self.beta > 0 and hasattr(self.model, 'kd_adapters'):
            # 수집된 Feature 가져오기
            s_feats = [self.student_features[i] for i in self.feature_layers if i in self.student_features]
            t_feats = [self.teacher_features[i] for i in self.teacher_feature_layers if i in self.teacher_features]
            
            if len(s_feats) > 0 and len(s_feats) == len(t_feats):
                # 어댑터 통과 (채널 맞추기)
                s_feats_adapted = self.model.kd_adapters(s_feats)
                
                for s_f, t_f in zip(s_feats_adapted, t_feats):
                    # [Fix] Cross-Resolution Distillation Support
                    # Teacher(1280px) -> 160x160, Student(960px) -> 120x120
                    # Student Feature를 Teacher 크기로 Upsampling하여 비교합니다.
                    if s_f.shape[-2:] != t_f.shape[-2:]:
                        s_f = F.interpolate(s_f, size=t_f.shape[-2:], mode='bilinear', align_corners=False)
                        
                    if self.feature_loss_type == 'cwd':
                        kd_loss_feature += self._cwd_loss(s_f, t_f)
                    else:
                        kd_loss_feature += F.mse_loss(s_f, t_f)
            
            # 다음 배치를 위해 초기화
            self.student_features.clear()
            self.teacher_features.clear()
            
        # Logit 저장소 초기화
        self.student_logits['box'].clear()
        self.student_logits['cls'].clear()
        self.teacher_logits['box'].clear()
        self.teacher_logits['cls'].clear()
        
        # 5. 최종 Loss 합산
        total_loss = loss + (self.alpha_box * kd_loss_box) + \
                     (self.alpha_cls * kd_loss_cls) + \
                     (self.beta * kd_loss_feature)
        
        # 6. 로깅용 값 저장
        self.loss_diagnostics = {
            'task_loss': loss.detach(),
            'kd_box': kd_loss_box.detach(),
            'kd_cls': kd_loss_cls.detach(),
            'kd_feature': kd_loss_feature.detach()
        }
        
        # [KD Debug] Loss Magnitude Monitoring (Console Print)
        # 매 배치를 출력하면 너무 많으니, 특정 확률이나 첫 배치만 출력하도록 하면 좋겠지만,
        # 사용자가 즉시 확인하길 원하므로 지금은 매번 출력하되 보기 좋게 포맷팅합니다.
        # (Trainer가 tqdm을 써서 깰 수 있지만, 중요한 정보임)
        # if torch.rand(1).item() < 0.01:
        # print(f" [KD Losses] Task: {loss.item():.4f} | Box: {kd_loss_box.item():.4f} | Cls: {kd_loss_cls.item():.4f} | Feat: {kd_loss_feature.item():.6f}")
        
        return total_loss, loss_items

# ==============================================================================
# 3. KD Trainer (학습기)
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
        kd_args = getattr(self, 'kd_args', {})
        t_path = kd_args.get('teacher_model')
        if not t_path:
            raise ValueError("KD Error: Teacher 모델 경로가 없습니다.")
            
        print(f"[KD] Teacher 모델 로드 중: {t_path}")
        teacher = YOLO(t_path)
        teacher.to(self.device)
        model.to(self.device)
        
        # [KD Auto-Detect] Teacher의 학습 해상도 감지
        try:
            if hasattr(teacher, 'ckpt') and 'train_args' in teacher.ckpt:
                t_imgsz = teacher.ckpt['train_args']['imgsz']
                print(f"[KD] Teacher 학습 해상도 감지됨: {t_imgsz}")
            else:
                # ckpt가 없는 경우 args 확인
                t_imgsz = getattr(teacher.model, 'args', {}).get('imgsz', 1280)
                print(f"[KD] Teacher 해상도 추정 (Fallback): {t_imgsz}")
        except Exception as e:
            print(f"[KD Warning] Teacher 해상도 감지 실패 (Default 1280): {e}")
            t_imgsz = 1280
            
        self.teacher_imgsz = t_imgsz
        
        # 3. Feature Layer 자동 감지 건너뛰기 (현재 하드코딩)
        # 안정성을 위해 P3, P4, P5 레이어 인덱스를 고정 사용합니다.
        print("[KD] 레이어 자동 감지를 생략하고 기본값을 사용합니다.")
        
        # YOLOv8/v11 공통 P3, P4, P5 인덱스
        self.verified_s_layers = [16, 19, 22]
        self.verified_t_layers = [16, 19, 22]
        
        # 채널 수 설정 (Student: Nano, Teacher: XLarge 기준)
        # [Fix] Inspection Script를 통해 확인된 정확한 채널 수로 수정
        s_channels = [64, 128, 256]   # Nano (Values Verified)
        t_channels = [384, 768, 768]  # XLarge (Values Verified: P3=384, P4=768, P5=768)
        
        # 4. Adapter 초기화 및 부착
        model.kd_adapters = FeatureAdapter(s_channels, t_channels).to(self.device)
        # Adapter 파라미터 학습 가능 설정
        for p in model.kd_adapters.parameters():
            p.requires_grad = True
            
        # [Fix] KDLoss 부모 클래스(v8DetectionLoss)가 model.args를 참조하므로
        # Loss 초기화 전에 args를 먼저 모델에 주입해야 합니다.
        if not hasattr(model, 'args'):
            model.args = self.args
            
        # 5. KD Loss 교체
        model.criterion = KDLoss(
            model, teacher.model, 
            alpha_box=kd_args.get('kd_alpha_box', 0.1),
            alpha_cls=kd_args.get('kd_alpha_cls', 0.5),
            beta=kd_args.get('kd_beta', 1.0),
            T=kd_args.get('kd_T', 4.0)
        )
        model.criterion.feature_loss_type = kd_args.get('kd_feature_type', 'mse')
        model.criterion.feature_layers = self.verified_s_layers
        model.criterion.teacher_feature_layers = self.verified_t_layers
        
        # 저장된 Args 주입 (Loss 내부 사용용)
        if not hasattr(model, 'args'):
            model.args = self.args
            
        # Hook 활성화
        model.criterion.restore_hooks()
        
        return model

    def save_model(self):
        """
        모델 저장 시 KD 전용 모듈(Adapter, Criterion)을 제거하여
        순수한 YOLO 포맷으로 저장되도록 합니다. (나중에 추론 시 에러 방지)
        """
        # --- 청소 함수 ---
        def clean(m):
            # Hook 제거
            for mod in m.modules():
                if hasattr(mod, '_forward_hooks'):
                    mod._forward_hooks.clear()
            # Adapter & Criterion 제거 및 백업
            adapters = getattr(m, 'kd_adapters', None)
            criterion = getattr(m, 'criterion', None)
            
            if hasattr(m, 'kd_adapters'): delattr(m, 'kd_adapters')
            if hasattr(m, 'criterion'): delattr(m, 'criterion')
            
            return adapters, criterion
            
        # --- 복구 함수 ---
        def restore(m, adp, crit):
            if adp: m.kd_adapters = adp
            if crit: 
                m.criterion = crit
                m.criterion.restore_hooks()

        # 1. Main 모델 청소
        m_adp, m_crit = clean(self.model)
        
        # 2. EMA 모델 청소 (있는 경우)
        e_adp, e_crit = None, None
        if hasattr(self, 'ema') and hasattr(self.ema, 'ema'):
            e_adp, e_crit = clean(self.ema.ema)
            
        # 3. 저장 실행
        super().save_model()
        
        # 4. 복구 (계속 학습을 위해)
        restore(self.model, m_adp, m_crit)
        if hasattr(self, 'ema') and hasattr(self.ema, 'ema'):
            restore(self.ema.ema, e_adp, e_crit)

# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Knowledge Distillation (Logit + Feature)')
    
    # 필수 경로
    parser.add_argument('--teacher', type=str, required=True, help='Teacher 모델 가중치 (.pt)')
    parser.add_argument('--student', type=str, required=True, help='Student 모델 가중치 (.pt)')
    parser.add_argument('--data', type=str, default='PCB_DATASET/data.yaml', help='데이터셋 설정 파일')
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=-1)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--project', type=str, default='runs/kd_feature')
    parser.add_argument('--name', type=str, default=None)
    
    # KD 하이퍼파라미터
    parser.add_argument('--alpha-box', type=float, default=0.1, help='Box Loss 가중치')
    parser.add_argument('--alpha-cls', type=float, default=0.5, help='Class Loss 가중치')
    parser.add_argument('--beta', type=float, default=1.0, help='Feature Loss 가중치')
    parser.add_argument('--temp', type=float, default=4.0, help='Temperature')
    parser.add_argument('--feature-loss-type', type=str, default='mse', choices=['mse', 'cwd'], help='Feature Loss 타입')
    
    # Optimizer 설정
    parser.add_argument('--optimizer', type=str, default='auto')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--cos-lr', action='store_true')

    args = parser.parse_args()
    args.project = os.path.abspath(args.project)

    # 실험 이름 자동 생성
    if args.name is None:
        s_name = os.path.splitext(os.path.basename(args.student))[0]
        t_name = os.path.splitext(os.path.basename(args.teacher))[0]
        args.name = f"KD_{s_name}_to_{t_name}"

    print(f"\n[KD] 학습 시작")
    print(f" - Teacher: {args.teacher}")
    print(f" - Student: {args.student}")
    print(f" - Optimizer: {args.optimizer} (LR: {args.lr0})")
    print(f" - Feature Loss: {args.feature_loss_type}")
    
    # Trainer 인자 준비
    train_args = vars(args)
    
    # KD 전용 인자 분리 (argparse의 dest 이름 기준)
    kd_keys = ['teacher', 'student', 'alpha_box', 'alpha_cls', 'beta', 'temp', 'feature_loss_type']
    
    kd_config = {
        'teacher_model': args.teacher,
        'kd_alpha_box': args.alpha_box,
        'kd_alpha_cls': args.alpha_cls,
        'kd_beta': args.beta,
        'kd_T': args.temp,
        'kd_feature_type': args.feature_loss_type
    }
    
    # YOLO Trainer에 전달할 오버라이드 인자 (KD 키 제외)
    # 얕은 복사로 원본 보존
    yolo_overrides = train_args.copy()
    yolo_overrides['model'] = args.student # [Fix] YOLO는 'model' 인자가 필수입니다.
    
    for k in kd_keys:
        if k in yolo_overrides:
            del yolo_overrides[k]
            
    # Trainer 실행
    trainer = KnowledgeDistillationTrainer(overrides=yolo_overrides)
    trainer.kd_args = kd_config
    trainer.train()
