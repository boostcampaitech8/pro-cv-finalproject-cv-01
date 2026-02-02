import os
from ultralytics import YOLO

class InferenceMgr:
    def __init__(self, model_path, config):
        self.model = YOLO(model_path)
        self.config = config
        # Output directory is determined at runtime via predict()

    def predict(self, source_txt, project="runs", name="inference"):
        """
        Predicts on images listed in source_txt (test set).
        """
        print(f"Running inference on {source_txt}...")
        
        # YOLOv8 can take a list of files, but let's read the txt file
        with open(source_txt, 'r') as f:
            files = [line.strip() for line in f.readlines() if line.strip()]
            
        # Run inference
        results = self.model.predict(
            source=files,
            imgsz=self.config['img_size'],
            conf=0.25, # Default confidence threshold
            save=True,
            save_txt=True,
            project=project,
            name=name,
            exist_ok=True
        )
        
        # Determine output directory (YOLO creates project/name)
        save_dir = os.path.join(project, name)
        
        # Generate CSV
        self.generate_csv(results, save_dir)
        
        return results

    def generate_csv(self, results, save_dir):
        """
        Parses YOLO Results objects and saves to CSV.
        Format: ImageID, Label, Confidence, xmin, ymin, xmax, ymax
        """
        import pandas as pd
        
        csv_data = []
        
        print("\nGenerating submission.csv...")
        for res in results:
            # Image ID (filename)
            image_id = os.path.basename(res.path)
            
            # Boxes
            boxes = res.boxes
            for i in range(len(boxes)):
                # Get class name
                cls_id = int(boxes.cls[i].item())
                label = res.names[cls_id]
                
                # Confidence
                conf = float(boxes.conf[i].item())
                
                # BBox (xyxy) - already in pixel coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                
                csv_data.append({
                    'ImageID': image_id,
                    'Label': label,
                    'Conf': conf,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2
                })
        
        if csv_data:
            os.makedirs(save_dir, exist_ok=True)
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(save_dir, 'submission.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV to: {csv_path}")
        else:
            print("No detections found. CSV not created.")

    def draw_bbox(self):
        # YOLOv8 'save=True' already draws bboxes.
        # This function is kept for custom visualization if needed later.
        pass
