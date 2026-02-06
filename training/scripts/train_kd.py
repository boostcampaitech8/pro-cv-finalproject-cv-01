import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO

class KDLoss(v8DetectionLoss):
    def __init__(self, model, teacher_model, alpha=0.5, T=4.0):
        super().__init__(model)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.T = T
        self.bce_kd = nn.BCEWithLogitsLoss(reduction='none') 
    
    def __call__(self, preds, batch):
        # 원본 Loss 계산
        loss, loss_items = super().__call__(preds, batch)
        
        # 1. Student Logits 추출
        student_scores = preds[1]['scores'] if isinstance(preds, tuple) else preds['scores']
        
        # 2. Teacher Logits 추출
        # Teacher 모델은 학습 모드가 아니므로, 일반적으로 전처리된 출력을 반환합니다.
        # 따라서 원시 Logit을 얻기 위해 입력을 직접 주입합니다.
        
        # 이미 __init__에서 teacher.model을 넘겨받았으므로 바로 사용
        teacher_module = self.teacher_model
        if hasattr(teacher_module, 'module'):
            teacher_module = teacher_module.module
            
        # 입력 데이터 타입을 Teacher 가중치 타입(Float/Half)에 맞춤 (AMP 대응)
        img = batch['img']
        teacher_dtype = next(teacher_module.parameters()).dtype
        if img.dtype != teacher_dtype:
            img = img.to(teacher_dtype)

        logits_t = None
        
        # Teacher 추론
        with torch.no_grad():
             output = teacher_module(img)
             if isinstance(output, tuple):
                 # output[1]은 'scores'를 포함한 예측 딕셔너리입니다.
                 teacher_preds = output[1]
                 logits_t = teacher_preds['scores']
             else:
                 print("경고: Teacher 모델이 튜플을 반환하지 않았습니다.")

        if logits_t is not None:
            # KD Loss 계산 (KL Divergence)
            # $L_{KD} = T^2 * KL( Softmax(Student/T), Softmax(Teacher/T) )$
            
            kd_loss = F.kl_div(
                F.log_softmax(student_scores / self.T, dim=-1),
                F.softmax(logits_t / self.T, dim=-1),
                reduction='batchmean'
            ) * (self.T ** 2)
            
            # 최종 Loss 합산
            loss = (1 - self.alpha) * loss + self.alpha * kd_loss
        
        return loss, loss_items

class KnowledgeDistillationTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        
        # main에서 주입된 kd_args 가져오기
        kd_args = getattr(self, 'kd_args', {})
        
        teacher_path = kd_args.get('teacher_model')
        if not teacher_path:
             raise ValueError("kd_args에 Teacher 모델 경로가 지정되지 않았습니다.")
             
        # Teacher 모델 로드
        if verbose:
            print(f"Teacher 모델 로드 중: {teacher_path}")
        # 상세 출력 억제 (verbose=False 시도 또는 context manager)
        try:
            teacher = YOLO(teacher_path)
        except Exception as e:
            print(f"Teacher 모델 로드 실패: {e}")
            raise e
        
        # Trainer가 설정한 장치로 이동
        teacher.to(self.device)

        # Loss 초기화 전 Student 모델도 동일 장치로 이동 (텐서 장치 불일치 방지)
        model.to(self.device)
        
        # Instantiate KDLoss
        alpha = kd_args.get('kd_alpha', 0.5)
        T = kd_args.get('kd_T', 4.0)
        
        # v8DetectionLoss가 초기화될 때 model.args가 필요하므로 수동으로 연결
        # (일반적으로 Trainer가 나중에 연결하지만, 우리는 지금 Loss를 초기화하므로 필요)
        model.args = self.args
        
        # 중요: teacher 객체 전체가 아닌 teacher.model(nn.Module)만 전달해야 함
        # YOLO 객체는 lock 등을 포함하여 pickle/deepcopy가 불가능할 수 있음 (EMA 저장 시 에러 발생)
        model.criterion = KDLoss(model, teacher.model, alpha=alpha, T=T)
        
        return model

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='YOLO Knowledge Distillation(지식 증류) 학습')
    parser.add_argument('--teacher', type=str, required=True, help='Teacher 모델 가중치 경로 (예: yolov11m_960.pt)')
    parser.add_argument('--student', type=str, required=True, help='Student 모델 가중치 경로 (예: yolov11n.pt) 또는 설정 파일')
    parser.add_argument('--data', type=str, default='PCB_DATASET/data.yaml', help='데이터셋 설정 파일 경로')
    parser.add_argument('--epochs', type=int, default=50, help='학습 Epoch 수')
    parser.add_argument('--imgsz', type=int, default=960, help='이미지 크기')
    parser.add_argument('--device', type=str, default='0', help='사용 장치 (0, 1, 0,1, cpu)')
    parser.add_argument('--project', type=str, default='runs/kd', help='프로젝트 저장 경로 (기본: runs/kd)')
    parser.add_argument('--name', type=str, default=None, help='실험 이름 (미지정 시 자동 생성)')
    parser.add_argument('--alpha', type=float, default=0.5, help='KD Loss 가중치 (alpha)')
    parser.add_argument('--temp', type=float, default=3.0, help='KD Temperature (T)')
    parser.add_argument('--batch', type=int, default=-1, help='배치 사이즈 (-1: 자동)')
    parser.add_argument('--patience', type=int, default=50, help='Early Stopping 인내값 (Epochs, 기본: 50)')
    
    args = parser.parse_args()
    
    # 상대 경로로 인한 중복 문제(runs/detect/runs/kd) 방지를 위해 절대 경로로 변환
    args.project = os.path.abspath(args.project)

    # 경로 확인
    if not os.path.exists(args.teacher):
        raise FileNotFoundError(f"Teacher 가중치를 찾을 수 없습니다: {args.teacher}")
    
    # 이름 자동 생성
    if args.name is None:
        def get_model_name(path):
            name = os.path.splitext(os.path.basename(path))[0]
            if name in ['best', 'last']:
                # 리모트/로컬 경로가 runs/exp_name/weights/best.pt 형태라고 가정하고 exp_name을 추출
                try:
                    # .. / weights / best.pt -> .. / weights -> .. -> exp_name
                    parent = os.path.dirname(os.path.dirname(path))
                    exp_name = os.path.basename(parent)
                    if exp_name: 
                        return exp_name
                except:
                    pass
            return name

        t_name = get_model_name(args.teacher)
        s_name = get_model_name(args.student)
        args.name = f"{s_name}_kd_t_{t_name}"

    print(f"KD 학습 시작:")
    print(f"  Teacher: {args.teacher}")
    print(f"  Student: {args.student}")
    print(f"  Data:    {args.data}")
    print(f"  ImgSz:   {args.imgsz}")

    # 학습 인자 딕셔너리
    train_args = dict(
        model=args.student,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        batch=args.batch,
        patience=args.patience,
        teacher_model=args.teacher,
        kd_alpha=args.alpha,
        kd_T=args.temp,
        exist_ok=True,
    )
    # 표준 YOLO 인자와 KD 전용 인자 분리
    kd_custom_keys = ['teacher_model', 'kd_alpha', 'kd_T']
    yolo_args = {k: v for k, v in train_args.items() if k not in kd_custom_keys}
    kd_args = {k: v for k, v in train_args.items() if k in kd_custom_keys}
    
    # YOLO 인자로 Trainer 초기화
    trainer = KnowledgeDistillationTrainer(overrides=yolo_args)
    
    # get_model에서 접근하도록 KD 인자 주입
    trainer.kd_args = kd_args
    
    trainer.train()
