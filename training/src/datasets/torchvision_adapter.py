import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from PIL import Image

class PCBTorchvisionDataset(Dataset):
    def __init__(self, txt_file, transforms=None):
        """
        Args:
            txt_file (str): Path to 'train.txt', 'val.txt', or 'test_images.txt'
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        with open(txt_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]
            
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load Image
        # Torchvision models usually expect PIL or Tensor 0-1
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        w, h = img.size
        
        # Load Label
        # images/Category/file.jpg -> labels/Category/file.txt
        label_path = img_path.replace('/images/', '/labels/').replace(os.path.splitext(img_path)[-1], '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = parts[1:]
                    
                    # Convert xywh_n -> xyxy
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id + 1) # Torchvision uses 0 for background, so class IDs shift +1
        
        
        target = {}
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # Check for degenerate boxes
            # Some boxes might be invalid (w or h < 0)
            valid_inds = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_inds]
            labels = labels[valid_inds]
            
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
            target["area"] = area
            target["iscrowd"] = iscrowd
        else:
            # Negative example
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["image_id"] = torch.tensor([idx])
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            # Check if transforms accepts (img, target)
            # v2 transforms usually return (img, target)
            # But standard checks are tricky. 
            # We assume user provides v2.Compose for both train/val if they use this class.
            
            # Wrap boxes in BoundingBoxes for v2 to recognize them
            # We need to import tv_tensors if available
            try:
                from torchvision import tv_tensors
                from torchvision.transforms import v2
                
                # Wrap Image
                img = tv_tensors.Image(img)
                
                # Wrap Target
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"]
                    # v2 requires format specification. YOLO format from adapter is xyxy?
                    # Adapter outputs xyxy in lines 53.
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        boxes, 
                        format="XYXY", 
                        canvas_size=(h, w) # Height, Width
                    )
                
                # Apply Transform
                img, target = self.transforms(img, target)
                
                # Unwrap if necessary? 
                # Torchvision models expect Tensor and Dict[Tensor].
                # tv_tensors.Image is a Tensor subclass, so it should be fine.
                # tv_tensors.BoundingBoxes is a Tensor subclass.
                
            except ImportError:
                # Fallback to standard transforms (Image Only)
                # Warning: invalid for spatial transforms
                img = self.transforms(img)
            except Exception as e:
                # Fallback or error
                # print(f"Transform failed: {e}")
                # Try image only
                img = self.transforms(img)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
