import os
import json
import cv2
from tqdm import tqdm


json_path = "./data/FSC147/annotation_FSC147_384.json"
img_dir = "./data/FSC147/images_384_VarV2/"
output_dir = "./data/FSC147/box/"

os.makedirs(output_dir, exist_ok=True)


print("Đang trích xuất các hộp mẫu (exemplars)...")
with open(json_path, 'r') as f:
    annotations = json.load(f)

for img_name, data in tqdm(annotations.items()):
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        continue
    
    image = cv2.imread(img_path)
    if image is None: continue
    
    bboxes = data.get('box_examples_coordinates', [])
    
    for i, box in enumerate(bboxes):
        x_coords = [int(pt[0]) for pt in box]
        y_coords = [int(pt[1]) for pt in box]
        
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
            save_name = f"{img_name.split('.')[0]}_box_{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, save_name), crop)

print(f"Hoàn thành! Các hộp mẫu đã được lưu vào: {output_dir}")