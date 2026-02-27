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
        if self.data_path.endswith('.yaml'):
            self.data_path = os.path.dirname(self.data_path)
        self.images_dir = os.path.join(self.data_path, 'images')
        self.annotations_dir = os.path.join(self.data_path, 'Annotations')
        self.labels_dir = os.path.join(self.data_path, 'labels') # YOLO 라벨 포맷 저장 디렉토리
        
        # 라벨 디렉토리가 존재하는지 확인하고 없으면 생성합니다.
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # 기본 검증(Validation)
        if not os.path.exists(self.images_dir):
            print(f"Warning: Images directory not found at {self.images_dir}")
        if not os.path.exists(self.annotations_dir):
            print(f"경고: {self.annotations_dir} 위치에서 Annotations 디렉토리를 찾을 수 없습니다.")
        
        self.classes = config['names']
        # XML 파일 간의 대소문자 차이 문제를 해결하기 위해 소문자 기반의 클래스 맵을 구성합니다.
        self.class_map = {name.lower(): i for i, name in enumerate(self.classes)}

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower() # 소문자로 정규화
            if name not in self.class_map:
                continue
                
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # YOLO 변환 규칙: 클래스 번호, 변경된 x_center, y_center, 넓이(width), 높이(height) (모두 0~1 정규화)
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            cls_id = self.class_map[name]
            objects.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
        return objects

    def convert_annotations(self):
        print("XML 형태의 정답 정보(Annotations)를 YOLO 포맷으로 변환합니다...")
        xml_files = glob.glob(os.path.join(self.annotations_dir, "*/*.xml"))
        
        if not xml_files:
            print(f"{self.annotations_dir} 디렉토리에서 XML 파일을 하나도 찾지 못했습니다. 기존 생성된 라벨 파일이 있는지 확인하십시오.")
            # Fallback 로직: XML이 없다면 이미지와 라벨이 이미 구비되어 있을 수 있습니다.
            # 하지만 현재 파이프라인 구조 상 여기에서 'valid_images' 리스트를 반환할 수 있어야 합니다.
            # 이미 변환된 데이터셋을 지원하려면 images 디렉토리를 직접 전체 탐색하는 로직이 추가로 필요합니다.
            pass
        
        valid_images = []
        
        for xml_file in tqdm(xml_files):
            # 1. 대상 XML과 매칭되는 이미지를 검색합니다.
            # 예상 파일 구조: Annotations/Category/file.xml -> images/Category/file.jpg
            # 주의: 확장자 케이스를 세밀하게 다루어야 합니다.
            
            # 현재 패키지 내 상대 경로 추출
            rel_path = os.path.relpath(xml_file, self.annotations_dir)
            category, filename = os.path.split(rel_path)
            basename = os.path.splitext(filename)[0]
            
            # 지원되는 여러 확장자를 시도하며 대응 이미지가 존재하는지 확인합니다.
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
            
            # 파싱 및 라벨 변환 및 저장
            yolo_lines = self.parse_xml(xml_file)
            
            # 대상 경로 생성 규칙: labels/Category/basename.txt
            label_subdir = os.path.join(self.labels_dir, category)
            os.makedirs(label_subdir, exist_ok=True)
            label_path = os.path.join(label_subdir, basename + ".txt")
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
                
        return valid_images

    def get_labels_from_txt(self, valid_images):
        """
        Stratification(계층화) 스플릿을 위해 각 이미지의 주 클래스를 추출합니다.
        하나의 주요 대표 클래스가 있다고 가정하거나, 파일 내 가장 첫 번째 라벨을 사용합니다.
        다중 클래스 이미지의 경우 가장 빈번한 클래스를 기준 삼아야 할 수도 있습니다.
        """
        labels = []
        for img_path in valid_images:
            # 라벨 경로 추론
            # 추론 예시: images/Category/file.jpg -> labels/Category/file.txt
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # 이미지 경로를 기반으로 카테고리 디렉토리 구조를 확인합니다.
            # 구조 전제: .../images/Category/file.jpg
            category = os.path.basename(os.path.dirname(img_path))
            
            label_path = os.path.join(self.labels_dir, category, basename + ".txt")
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # 찾은 첫 번째 라벨 ID(클래스 번호) 채택
                        cls_id = int(lines[0].split()[0])
                        labels.append(cls_id)
                    else:
                        labels.append(-1) # 파일은 있으나 빈 객체인 경우
            else:
                labels.append(-1)
        return labels

    def prepare(self):
        """
        메인 데이터 파이프라인 준비부:
        1. 주어진 XML 정보를 찾아 YOLO 포맷 변환 (convert_annotations)
        2. Test 셋 10% 분리 (클래스 비율 맞춤 Stratified)
        3. Train/Val 셋 분할 (클래스 비율 맞춤 Stratified)
        """
        # 1. 변환 과정 수행
        all_images = self.convert_annotations()
        # 데이터 편향 방지를 위해 파일 기반 라벨링 클래스 파악
        all_labels = self.get_labels_from_txt(all_images)
        
        # 빈 배경 이미지(-1) 데이터 처리 로직을 원한다면 여기에 추가 구성 가능합니다.
        # 현재 코드 상에서는 그대로 보존하되 -1로 표기합니다. 
        
        X = np.array(all_images)
        y = np.array(all_labels)
        
        if len(X) == 0:
            raise FileNotFoundError(
                f"{self.data_path} 영역에서 유효한 이미지를 식별하지 못했습니다. "
                "하위에 'images' 폴더와 'Annotations' 폴더가 올바른 구조로 있는지 확인해주세요. "
                f"현재 탐색 시도 경로: {os.path.abspath(self.data_path)}"
            )

        # 2. 비율 기반 데이터 분할 및 텍스트 리스트(txt) 저장 처리
        self._split_and_save(X, y)
        
        # 통합 환경 관리용 data.yaml 생성하기
        yaml_path = self.create_data_yaml(
            os.path.join(self.data_path, 'train.txt'), 
            os.path.join(self.data_path, 'val.txt')
        )
        return yaml_path

    def _split_and_save(self, X, y, train_save_path=None):
        """
        기본 7:2:1 (Train:Val:Test) 비율로 Stratified Split 진행.
        (K-Fold 분할은 수행하지 않고 단일 시퀀스로 분할합니다.)
        """
        # 1단계: Test 셋 분할 (설정된 test_size 할당, 기본 클래스 비율 배분)
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], stratify=y, random_state=42
            )
        except ValueError:
            print("경고: 클래스 샘플 수가 부족해 Stratified Split이 실패했습니다. 무작위 분할 모드로 전환합니다.")
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], random_state=42
            )
        
        # 2단계: 남은 데이터(Train+Val) 중 Validation 셋 분할
        # 전체 데이터 중 val_size (예: 20%)를 차지해야 하므로
        # 현재 남은 비중(예: 90%) 속에서 2/9 의 할당분을 가져옵니다.
        val_ratio = self.config['val_size'] / (1.0 - self.config['test_size'])
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=42
            )
        except ValueError:
             X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, random_state=42
            )

        # 결과 리스트 저장 (텍스트 파일 형태)
        with open(os.path.join(self.data_path, 'test_images.txt'), 'w') as f:
            f.write('\n'.join(X_test))
            
        # train 저장 경로 설정 (기본값: data_path/train.txt)
        if train_save_path is None:
            train_save_path = os.path.join(self.data_path, 'train.txt')
            
        with open(train_save_path, 'w') as f:
            f.write('\n'.join(X_train))
            
        with open(os.path.join(self.data_path, 'val.txt'), 'w') as f:
            f.write('\n'.join(X_val))
            
        print(f"총 분석 이미지 수량: {len(X)}")
        print(f"학습셋(Train) 배치 수량: {len(X_train)} ({len(X_train)/len(X):.2%})")
        print(f"검증셋(Val) 배치 수량:   {len(X_val)} ({len(X_val)/len(X):.2%})")
        print(f"운영 테스트(Test) 수량:  {len(X_test)} ({len(X_test)/len(X):.2%})")
        
        return X_train

    def create_data_yaml(self, train_txt, val_txt):
        """
        YOLOv8 호환을 위한 데이터셋 정의 파일(data.yaml)을 생성합니다.
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
    # 내부 테스트 용도 
    pass

def get_dataset(config):
    """
    모듈 외부에서 PCBDataset 인스턴스를 요청하여 생성(Factory)하기 위한 래핑 객체입니다.
    """
    return PCBDataset(config)
