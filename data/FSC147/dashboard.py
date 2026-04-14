import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from glob import glob
from tqdm import tqdm # Cài đặt bằng: pip install tqdm (để hiện thanh tiến trình)

# 1. CẤU HÌNH ĐƯỜNG DẪN
folder_npy = r"D:/UIT Files/PROJECTS/Zero-Shot Object Counting/FSC147_384_V2/gt_density_map_adaptive_384_VarV2"
folder_img = r"D:/UIT Files/PROJECTS/Zero-Shot Object Counting/FSC147_384_V2/images_384_VarV2"
output_dir = r"D:/UIT Files/PROJECTS/Zero-Shot Object Counting/comparison_results"

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# 2. LẤY DANH SÁCH FILE
npy_files = glob(os.path.join(folder_npy, "*.npy"))
print(f"Tìm thấy {len(npy_files)} file dữ liệu.")

# 3. VÒNG LẶP TỰ ĐỘNG
for npy_path in tqdm(npy_files):
    file_name = os.path.basename(npy_path)
    file_id = os.path.splitext(file_name)[0]
    
    # Tìm ảnh tương ứng (thử cả .jpg và .png)
    img_path = os.path.join(folder_img, f"{file_id}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(folder_img, f"{file_id}.png")

    # Load dữ liệu
    density_map = np.load(npy_path)
    count = np.sum(density_map)

    # Vẽ dashboard
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ảnh gốc
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img)
    else:
        ax1.text(0.5, 0.5, 'Image Not Found', ha='center')
    ax1.set_title(f"Original: {file_id}")
    ax1.axis('off')

    # Density Map
    im2 = ax2.imshow(density_map, cmap='jet')
    ax2.set_title(f"Density (Count: {count:.2f})")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Lưu kết quả
    save_path = os.path.join(output_dir, f"compare_{file_id}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close() # Đóng plot để giải phóng bộ nhớ RAM

print(f"\nHoàn tất! Kết quả đã lưu tại: {output_dir}")