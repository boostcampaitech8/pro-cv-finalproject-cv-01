import os
import sys
import yaml
import shutil
import random
import boto3
from pathlib import Path
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import config (assuming similar structure to API config or standalone)
# For simplicity, we'll try to load from a yaml or env, but here we hardcode or use explicit args
# Ideally, reuse the same config object used in API.
try:
    from config import settings
except ImportError:
    # Fallback/Dummy settings if not running in same env as API
    class Settings:
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
        S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "pcb-data-storage")
        DATASET_DIR = os.path.join(project_root, "PCB_DATASET")
    settings = Settings()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

def sync_refined_data(local_dataset_msg_dir: str):
    """
    Downloads new files from S3 refined/images and refined/labels
    Returns a list of new image filenames (stem).
    """
    bucket = settings.S3_BUCKET_NAME
    prefix = "refined/"
    
    # Ensure directories exist
    images_dir = Path(local_dataset_msg_dir) / "images"
    labels_dir = Path(local_dataset_msg_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Syncing from s3://{bucket}/{prefix}...")
    
    # List all objects in refined/
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    new_files = []
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)
            
            # Skip directories
            if not filename: 
                continue

            # Check if it's image or label
            if key.startswith("refined/images/") and (filename.endswith('.jpg') or filename.endswith('.png')):
                target_path = images_dir / filename
                if not target_path.exists(): # Only download if not exists
                    s3_client.download_file(bucket, key, str(target_path))
                    new_files.append(target_path.stem)
                    # Also try to download corresponding label
                    label_key = key.replace("images/", "labels/").replace(Path(key).suffix, ".txt")
                    label_path = labels_dir / (Path(filename).stem + ".txt")
                    try:
                        s3_client.download_file(bucket, label_key, str(label_path))
                    except Exception:
                        print(f"⚠️ Warning: Missing label for {filename}")
            
    print(f"✅ Downloaded {len(new_files)} new images.")
    return new_files

def load_existing_splits(dataset_dir):
    """
    Reads train.txt and val.txt to identify already registered images.
    Returns: set of absolute paths (or stems if consistent)
    """
    existing_images = set()
    for split in ["train.txt", "val.txt", "test_images.txt"]:
        path = Path(dataset_dir) / split
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    # Store as absolute path for reliable comparison
                    existing_images.add(line.strip())
    return existing_images

def create_transient_datasets(dataset_dir, new_stems):
    """
    Creates temporary dataset files for retraining:
    - train_retrain.txt = train.txt + new_train (90%)
    - val_retrain.txt = val.txt + new_val (10%)
    - data_retrain.yaml = Points to these new files
    """
    # 1. Load Original Splits
    train_orig = []
    val_orig = []
    
    if (Path(dataset_dir) / "train.txt").exists():
        with open(Path(dataset_dir) / "train.txt", "r") as f:
            train_orig = [line.strip() for line in f if line.strip()]
            
    if (Path(dataset_dir) / "val.txt").exists():
        with open(Path(dataset_dir) / "val.txt", "r") as f:
            val_orig = [line.strip() for line in f if line.strip()]

    print(f"📉 Base Dataset: {len(train_orig)} Train, {len(val_orig)} Val")

    # 2. Split New Data (9:1)
    if new_stems:
        random.shuffle(new_stems)
        split_idx = int(len(new_stems) * 0.9)
        new_train_stems = new_stems[:split_idx]
        new_val_stems = new_stems[split_idx:]
        
        print(f"🆕 New Data: {len(new_train_stems)} Train, {len(new_val_stems)} Val")
    else:
        new_train_stems = []
        new_val_stems = []
        print("ℹ️  No new data found. Using base dataset only.")

    # Helper to convert stems to paths
    def stems_to_paths(stems):
        paths = []
        for s in stems:
            # Search for the stem in all downloaded images to get the correct path (including any subdirectories if they exist, or the downloaded root)
            found = list(Path(dataset_dir).rglob(f"{s}.jpg"))
            if found:
                paths.append(str(found[0].absolute()))
        return paths

    new_train_paths = stems_to_paths(new_train_stems)
    new_val_paths = stems_to_paths(new_val_stems)

    # 3. Combine
    train_final = train_orig + new_train_paths
    val_final = val_orig + new_val_paths

    # 4. Write Transient Files
    retrain_train_path = Path(dataset_dir) / "train_retrain.txt"
    retrain_val_path = Path(dataset_dir) / "val_retrain.txt"

    with open(retrain_train_path, "w") as f:
        f.write("\n".join(train_final))
    
    with open(retrain_val_path, "w") as f:
        f.write("\n".join(val_final))

    print(f"✅ Created Transient Splits: {len(train_final)} Train, {len(val_final)} Val")
    print(f"   -> {retrain_train_path}")
    print(f"   -> {retrain_val_path}")

    # 5. Create Transient Data YAML
    # We need to read the original data.yaml to get class names
    orig_yaml_path = Path(dataset_dir) / "data.yaml"
    names = {}
    nc = 0
    
    if orig_yaml_path.exists():
        with open(orig_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            names = data_config.get('names', {})
            nc = data_config.get('nc', len(names))
    else:
        # Fallback: PCB defect class names (from config.settings CLASS_ID_MAP)
        names = {
            0: "missing_hole",
            1: "mouse_bite",
            2: "open_circuit",
            3: "short",
            4: "spur",
            5: "spurious_copper"
        }
        nc = 6
        print("⚠️  data.yaml not found. Using default PCB defect class names.")
    
    retrain_yaml_content = {
        'path': str(dataset_dir),
        'train': str(retrain_train_path.absolute()),
        'val': str(retrain_val_path.absolute()),
        'nc': nc,
        'names': names
    }

    retrain_yaml_path = Path(dataset_dir) / "data_retrain.yaml"
    with open(retrain_yaml_path, 'w') as f:
        yaml.dump(retrain_yaml_content, f)
        
    print(f"📄 Created Transient Config: {retrain_yaml_path}")
    return str(retrain_yaml_path)

def main():
    dataset_dir = settings.DATASET_DIR
    
    # 1. Sync (Downloads files, returns all new stems found on disk that aren't in base)
    # Modified Logic: existing 'sync_refined_data' downloads files.
    # We need to identify which of these are actually "new" to the dataset.
    
    # Run sync first to ensure files are local
    downloaded_stems = sync_refined_data(dataset_dir) # This downloads everything new from S3
    
    # Now SCAN disk to find everything available
    images_dir = Path(dataset_dir) / "images"
    all_stems = {f.stem for f in images_dir.rglob("*.jpg")}
    
    # Load what we already have in train.txt/val.txt
    # We need to extract stems from the paths in txt files
    existing_paths = load_existing_splits(dataset_dir)
    existing_stems = {Path(p).stem for p in existing_paths}
    
    # Identify truly new stems (On disk BUT NOT in existing splits)
    new_stems = list(all_stems - existing_stems)
    
    print(f"🔎 Scanning: Found {len(all_stems)} images on disk, {len(existing_stems)} already in splits.")
    print(f"✨ Identified {len(new_stems)} unassigned images.")

    # 2. Create Transient Files
    yaml_path = create_transient_datasets(dataset_dir, new_stems)
    
    # Output the yaml path for Airflow to capture (via XCom or stdout lines)
    print(f"::set-output name=data_yaml::{yaml_path}")

if __name__ == "__main__":
    main()
