import os
import json
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

DATA_DIR = "./data/FSC147/images_384_VarV2/"
POS_JSON = "./data/FSC147/annotation_FSC147_pos.json"
NEG_JSON = "./data/FSC147/annotation_FSC147_neg.json"

OUT_DIR_POS = "./data/FSC147/Masks_Pos/"
OUT_DIR_NEG = "./data/FSC147/Masks_Neg/"

os.makedirs(OUT_DIR_POS, exist_ok=True)
os.makedirs(OUT_DIR_NEG, exist_ok=True)

print("Đang khởi tạo SAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"


sam_checkpoint = "./sam_vit_b_01ec64.pth" 
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


def process_masks(json_path, output_dir):
    print(f"\n--- ĐANG XỬ LÝ FILE: {json_path} ---")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for img_name, img_data in tqdm(data.items(), desc="Cắt Mask"):
        img_path = os.path.join(DATA_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        # Đọc ảnh bằng OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]
        
        # Tạo mask nền đen 100%
        final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        
        # Nạp ảnh vào bộ nhớ của SAM
        predictor.set_image(image)
        
        boxes = img_data.get("box_examples_coordinates",[])
        for box in boxes:
            if len(box) == 0: continue
            
            # Format trong JSON: [[x1,y1],[x1,y2], [x2,y2], [x2,y1]] -> Đổi thành[x_min, y_min, x_max, y_max]
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            
            # Tính giới hạn an toàn để không vượt quá viền ảnh
            x_min = max(0, int(min(x_coords)))
            x_max = min(w_img, int(max(x_coords)))
            y_min = max(0, int(min(y_coords)))
            y_max = min(h_img, int(max(y_coords)))
            
            # Đưa Bounding box vào SAM
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            # Gộp mask của hộp này vào mask tổng (Dùng toán tử OR)
            mask = masks[0].astype(np.uint8) 
            final_mask = np.logical_or(final_mask, mask).astype(np.uint8)
        
        # Nhân với 255 để: Điểm có mask = 255 (Màu trắng), Nền = 0 (Màu đen)
        final_mask = final_mask * 255
        
        # Lưu thành file .png (Tuyệt đối không dùng .jpg vì sẽ bị mờ viền)
        out_path = os.path.join(output_dir, img_name.replace(".jpg", ".png"))
        cv2.imwrite(out_path, final_mask)


process_masks(POS_JSON, OUT_DIR_POS)
process_masks(NEG_JSON, OUT_DIR_NEG)

print("\nTOÀN BỘ DATASET ĐÃ ĐƯỢC SAM CẮT THÀNH MASK!")