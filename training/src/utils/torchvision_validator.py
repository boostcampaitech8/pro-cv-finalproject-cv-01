from ultralytics.models.yolo.detect import DetectionValidator
import torch

class TorchvisionValidator(DetectionValidator):
    """
    Subclass of Ultralytics DetectionValidator to handle Torchvision models.
    Since TorchvisionWrapper already outputs NMS-processed boxes in [N, 6] format,
    we override postprocess to bypass YOLO's NMS logic.
    """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        # Ultralytics 8.4.5: __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None)
        # pbar removed from signature
        super().__init__(dataloader, save_dir, args=args, _callbacks=_callbacks)
        self.args.task = 'detect'

    def postprocess(self, preds):
        """
        Pass-through since TorchvisionWrapper returns already processed boxes.
        Standard YOLO postprocess does NMS here and returns a LIST OF DICTS.
        We must match that format: [{'bboxes': ..., 'conf': ..., 'cls': ...}, ...]
        Args:
            preds (list[torch.Tensor]): List of (N, 6) tensors.
        Returns:
            list[dict[str, torch.Tensor]]: Processed predictions in Ultralytics format.
        """
        output = []
        for pred in preds:
            # pred is (N, 6) [x1, y1, x2, y2, conf, cls]
            if pred is None or len(pred) == 0:
                 output.append({"bboxes": torch.zeros((0, 4), device=self.device), 
                                "conf": torch.zeros((0), device=self.device), 
                                "cls": torch.zeros((0), device=self.device)})
                 continue
            
            output.append({
                "bboxes": pred[:, :4],
                "conf": pred[:, 4],
                "cls": pred[:, 5]
            })
        return output

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        # Standard Init
        super().init_metrics(model)
        # Ensure names are synced if model wrapper didn't have them set correctly
        if hasattr(model, 'names'):
            self.names = model.names
            self.metrics.names = self.names
            self.nc = len(model.names)
