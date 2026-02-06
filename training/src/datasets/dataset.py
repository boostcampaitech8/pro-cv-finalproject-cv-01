import os
import glob
import random
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import numpy as np

class PCBDataset:
    def __init__(self, config):
        self.config = config
        self.data_path = config['data_path']
        self.images_dir = os.path.join(self.data_path, 'images')
        self.annotations_dir = os.path.join(self.data_path, 'Annotations')
        self.labels_dir = os.path.join(self.data_path, 'labels') # YOLO labels
        
        # Ensure label directory exists
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Validation
        if not os.path.exists(self.images_dir):
            print(f"Warning: Images directory not found at {self.images_dir}")
        if not os.path.exists(self.annotations_dir):
            print(f"Warning: Annotations directory not found at {self.annotations_dir}")
        
        self.classes = config['names']
        # Map lowercase name to ID to handle XML case differences
        self.class_map = {name.lower(): i for i, name in enumerate(self.classes)}

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower() # Normalize to lowercase
            if name not in self.class_map:
                continue
                
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # YOLO Format: class x_center y_center width height (normalized)
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            cls_id = self.class_map[name]
            objects.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
        return objects

    def convert_annotations(self):
        print("Converting XML annotations to YOLO format...")
        xml_files = glob.glob(os.path.join(self.annotations_dir, "*/*.xml"))
        
        if not xml_files:
            print(f"No XML files found in {self.annotations_dir}. Checking for existing labels...")
            # Fallback: if no XMLs, maybe we just have images and labels already?
            # But the current pipeline relies on returning 'valid_images' from this process.
            # If we want to support pre-converted data, we'd need to scan images directory directly.
            pass
        
        valid_images = []
        
        for xml_file in tqdm(xml_files):
            # Find corresponding image
            # Structure: Annotations/Category/file.xml -> images/Category/file.jpg
            # Note: We need to handle file extensions carefully.
            
            # Extract relative path components
            rel_path = os.path.relpath(xml_file, self.annotations_dir)
            category, filename = os.path.split(rel_path)
            basename = os.path.splitext(filename)[0]
            
            # Check for image with supported extensions
            image_found = False
            image_path = ""
            for ext in ['.jpg', '.JPG', '.png', '.jpeg']:
                chk_path = os.path.join(self.images_dir, category, basename + ext)
                if os.path.exists(chk_path):
                    image_path = chk_path
                    image_found = True
                    break
            
            if not image_found:
                continue
                
            valid_images.append(image_path)
            
            # Convert and save label
            yolo_lines = self.parse_xml(xml_file)
            
            # Label path: labels/Category/basename.txt
            label_subdir = os.path.join(self.labels_dir, category)
            os.makedirs(label_subdir, exist_ok=True)
            label_path = os.path.join(label_subdir, basename + ".txt")
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
                
        return valid_images

    def get_labels_from_txt(self, valid_images):
        """
        Extracts the primary class for each image for stratification.
        Assumes single-label-dominant or uses the first label.
        For multi-label, we might stratify by the most frequent class.
        """
        labels = []
        for img_path in valid_images:
            # Infer label path
            # images/Category/file.jpg -> labels/Category/file.txt
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # We need the relative category folder from the image path
            # Assuming structure: .../images/Category/file.jpg
            category = os.path.basename(os.path.dirname(img_path))
            
            label_path = os.path.join(self.labels_dir, category, basename + ".txt")
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Use the first class ID found
                        cls_id = int(lines[0].split()[0])
                        labels.append(cls_id)
                    else:
                        labels.append(-1) # Empty
            else:
                labels.append(-1)
        return labels

    def prepare(self):
        """
        Main pipeline:
        1. Convert XML -> YOLO
        2. Split Test set (10%) - Stratified
        3. Split K-Folds (90%) - Stratified
        """
        # 1. Convert Annotations
        all_images = self.convert_annotations()
        # Get labels for stratification
        all_labels = self.get_labels_from_txt(all_images)
        
        # Filter out empty label images if desired, or handle them.
        # Here we keep them but warn if label is -1
        
        X = np.array(all_images)
        y = np.array(all_labels)
        
        if len(X) == 0:
            raise FileNotFoundError(
                f"No valid images found in {self.data_path}. "
                "Please ensure 'images' and 'Annotations' directories exist and contain data. "
                f"Current search path: {os.path.abspath(self.data_path)}"
            )

        # 2. Split and Save (Stratified)
        self._split_and_save(X, y)
        
        # Create single data.yaml
        yaml_path = self.create_data_yaml(
            os.path.join(self.data_path, 'train.txt'), 
            os.path.join(self.data_path, 'val.txt')
        )
        return yaml_path

    def _split_and_save(self, X, y, train_save_path=None):
        """
        Performs 7:2:1 Stratified Split and saves txt files.
        Only generates a single split (No K-Fold).
        """
        # 2. Split Test Set (Stratified)
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], stratify=y, random_state=42
            )
        except ValueError:
            print("Warning: Stratified split failed (too few samples?). Fallback to random split.")
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], random_state=42
            )
        
        # 3. Split Validation Set (Stratified)
        # We need 20% of total for Val. 
        # Remaining is 90% (Train+Val). Val is 2/9 of that.
        val_ratio = self.config['val_size'] / (1.0 - self.config['test_size'])
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=42
            )
        except ValueError:
             X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, random_state=42
            )

        # Save splits
        with open(os.path.join(self.data_path, 'test_images.txt'), 'w') as f:
            f.write('\n'.join(X_test))
            
        # Determine train save path (default to train.txt, or custom if provided)
        if train_save_path is None:
            train_save_path = os.path.join(self.data_path, 'train.txt')
            
        with open(train_save_path, 'w') as f:
            f.write('\n'.join(X_train))
            
        with open(os.path.join(self.data_path, 'val.txt'), 'w') as f:
            f.write('\n'.join(X_val))
            
        print(f"Total images: {len(X)}")
        print(f"Train images: {len(X_train)} ({len(X_train)/len(X):.2%})")
        print(f"Val images:   {len(X_val)} ({len(X_val)/len(X):.2%})")
        print(f"Test images:  {len(X_test)} ({len(X_test)/len(X):.2%})")
        
        return X_train

    def create_data_yaml(self, train_txt, val_txt):
        """
        Creates data.yaml for YOLOv8
        """
        yaml_content = {
            'path': self.data_path,
            'train': train_txt,
            'val': val_txt,
            'test': os.path.join(self.data_path, 'test_images.txt'),
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        yaml_path = os.path.join(self.data_path, f'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
            
        return yaml_path

if __name__ == "__main__":
    # Test script
    pass

def get_dataset(config):
    """
    Factory function to return the dataset instance.
    """
    return PCBDataset(config)
