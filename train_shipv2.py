import os
from pathlib import Path
import yaml
import cv2
import numpy as np
from ultralytics import YOLO

# 1️⃣ 경로 설정
project_root = Path('./whole_to_12cls')
dataset_root = Path('../datasets/shipdatasets/matched_sup')  # 이미지와 라벨 폴더 구조: images/train, labels/train 등
project_root.mkdir(exist_ok=True)

# 2️⃣ 클래스 정의
class_names = [
    'Motor_boat', 'Sail_boat', 'Tugboat', 'Barge',
    'Fishing_boat', 'Ferry', 'Container_Ship', 'Oil_tanker',
    'Drill_ship', 'Warship', 'Submarine', 'Others'
]
num_classes = len(class_names)

# 3️⃣ data.yaml 저장
data_yaml = {
    'path': str(dataset_root),
    'train': 'images/train',
    'val': 'images/val',
    'nc': num_classes,
    'names': class_names
}
with open(project_root / 'data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

# 4️⃣ model.yaml 생성 (선택사항: default는 yolo11n-obb.pt에서 자동 감지됨)

# 5️⃣ OBB 라벨 시각화 함수
def visualize_obb(image_path, label_path, save_path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls, cx, cy, bw, bh, angle = map(float, line.strip().split())
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            rect = ((cx, cy), (bw, bh), angle)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            cv2.putText(img, class_names[int(cls)], (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite(str(save_path), img)

# 예시 시각화 (원하는 이미지와 라벨 경로 넣기)
# visualize_obb(dataset_root/'images/val/img001.jpg', dataset_root/'labels/val/img001.txt', project_root/'vis001.jpg')

# 6️⃣ YOLOv11-OBB 모델 로드 및 fine-tune
model = YOLO('Whole_ship_pretrain/1024res/weights/last.pt')

        #Whole_ship_pretrain/1024res/weights/last.pt')  # 또는 'yolo11x-obb.pt'
#model = YOLO('ship12_experiment_cls0/ship12-finetune-obb3/weights/last.pt')
# 현재 부른건0 CLASS로 학습한 112에폭 mAP 0.498짜리

model.train(
    data=str(project_root / 'data.yaml'),
    epochs=600,
    imgsz=1024,
    batch=4,
    #lr0=0.01,
    #optimizer='SGD',
    single_cls=False, 
    freeze=22, 
    #multi_scale=True,
    # overlap_mask=True,
    #mask_ratio=4, 
    #dropout=0.0,
    #augment=True, 
    name='1024res_freeze22',
    project=str(project_root),
    pretrained='Whole_ship_pretrain/1024res/weights/last.pt',
    close_mosaic=10,
    device=0,
    workers=8,
    verbose=True,
)


#optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.1, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None
