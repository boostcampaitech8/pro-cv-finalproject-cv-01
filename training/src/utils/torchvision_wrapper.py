import torch
import torch.nn as nn

class TorchvisionModelWrapper(nn.Module):
    """
    Wraps a Torchvision Detection Model to mimic Ultralytics YOLO interface.
    This allows using Ultralytics DetectionValidator for identical metric calculation.
    """
    def __init__(self, model, class_map=None):
        super().__init__()
        self.model = model
        self.class_map = class_map # Optional mapping if needed
        self.args = None # Placeholder for Validator compatibility
        self.names = {i: str(i) for i in range(80)} # Default names, overwritten by validator
        self.stride = torch.tensor([32]) # Fake stride for consistency checks (Must be Tensor for .max())
        self.pt = True
        self.fp16 = False # Torchvision models usually FP32 by default
        self.yaml = {} # Dummy yaml dict

    def fuse(self, verbose=True):
        """Dummy fuse method for Ultralytics compatibility."""
        return self

    def forward(self, x, augment=False, visualize=False, embed=None):
        """
        Args:
            x (torch.Tensor): Input batch [B, 3, H, W] (0-1 Normalized usually)
        Returns:
            list[torch.Tensor]: [B] list of (N, 6) detections (x1, y1, x2, y2, conf, cls)
        """
        # Torchvision models expect list of tensors
        images = [img for img in x]
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            
        formatted_preds = []
        for out in outputs:
            # out is dict: {'boxes': ..., 'scores': ..., 'labels': ...}
            boxes = out['boxes']
            scores = out['scores']
            labels = out['labels']
            
            # Diagnostic (Print once per epoch effectively or randomly)
            if len(scores) > 0 and torch.rand(1).item() < 0.01:
                print(f"\n[DIAGNOSTIC] Max Score: {scores.max().item():.4f} | Top Classes: {labels[:5].tolist()}")
            
            # Filter low confidence? Usually Validator postprocess does, but we can do minimal
            # Torchvision outputs usually have many boxes.
            
            # Shift labels: TV 1 -> YOLO 0
            labels = labels - 1
            
            # Stack: x1, y1, x2, y2, conf, cls
            if len(boxes) > 0:
                pred = torch.cat([
                    boxes, 
                    scores.unsqueeze(1), 
                    labels.float().unsqueeze(1)
                ], dim=1)
            else:
                pred = torch.zeros((0, 6), device=x.device)
                
            formatted_preds.append(pred)
            
        # Return as list (Validator expects list for raw preds?)
        # Actually DetectionValidator.postprocess expects a single Tensor if it's YOLO Output (concat)
        # OR a list if simplified.
        # But wait, BaseValidator calls `preds = model(batch['img'])`.
        # Then `preds = self.postprocess(preds)`.
        # YOLO postprocess expects (B, 84, 8400) usually.
        # If I return already NMS'd boxes, I should override postprocess in the Validator too?
        # OR I can return a format that my custom postprocess handles.
        
        # Let's return this list of (N, 6).
        # We will subclass Validator to skip NMS if preds are already (N, 6).
        return formatted_preds
