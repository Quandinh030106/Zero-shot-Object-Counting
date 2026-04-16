import json
import random
import copy
import os

input_json = "./data/FSC147/annotation_FSC147_384.json"
pos_json = "./data/FSC147/annotation_FSC147_pos.json"
neg_json = "./data/FSC147/annotation_FSC147_neg.json"

print("Đang đọc dữ liệu gốc...")
if not os.path.exists(input_json):
    print(f"LỖI: Không tìm thấy file {input_json}")
    exit()

with open(input_json, 'r') as f:
    data = json.load(f)

with open(pos_json, 'w') as f:
    json.dump(data, f, indent=4)
print("Đã tạo xong file POSITIVE.")


print("Đang sinh các hộp Negative...")
neg_data = copy.deepcopy(data)

for img_name, info in neg_data.items():
    points = info.get("points",[])
    gt_boxes = info.get("box_examples_coordinates",[])
    
    if not gt_boxes:
        continue
        
    # Tính kích thước trung bình của hộp mẫu để làm hộp negative
    w_box = abs(gt_boxes[0][2][0] - gt_boxes[0][0][0])
    h_box = abs(gt_boxes[0][2][1] - gt_boxes[0][0][1])
    
    # Tránh trường hợp hộp bị lỗi kích thước = 0
    if w_box <= 0: w_box = 20
    if h_box <= 0: h_box = 20

    # Lấy kích thước ảnh (Mặc định bản resize là 384x384)
    W = 384
    H = 384
    
    neg_boxes =[]
    attempts = 0
    
    while len(neg_boxes) < 3 and attempts < 100:
        attempts += 1
        
        # Lấy random tọa độ góc trên bên trái
        x1 = random.uniform(0, max(1, W - w_box))
        y1 = random.uniform(0, max(1, H - h_box))
        x2 = x1 + w_box
        y2 = y1 + h_box
        
        # Kiểm tra xem hộp này có vô tình đè lên point nào không
        is_overlap = False
        for p in points:
            px, py = p[0], p[1]
            if x1 <= px <= x2 and y1 <= py <= y2:
                is_overlap = True
                break
        
        if not is_overlap:
            new_box = [[x1, y1], [x1, y2],[x2, y2], [x2, y1]]
            neg_boxes.append(new_box)
    
    while len(neg_boxes) < 3:
        neg_boxes.append([[0,0], [0,5], [5,5], [5,0]])
        
    # Cập nhật hộp negative vào dữ liệu
    neg_data[img_name]["box_examples_coordinates"] = neg_boxes

# Ghi ra file JSON
with open(neg_json, 'w') as f:
    json.dump(neg_data, f, indent=4)
    
print("Đã tạo xong file NEGATIVE.")