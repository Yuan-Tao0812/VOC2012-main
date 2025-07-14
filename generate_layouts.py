import os
import cv2
import numpy as np
from tqdm import tqdm

def draw_yolo_boxes(label_file, img_size=(512, 512)):
    """将一个YOLO标签文件转换为单通道黑底白框图像，带边界检查"""
    img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)  # 单通道黑底图

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, w, h = map(float, parts)
            x1 = max(0, int((x_center - w / 2) * img_size[0]))
            y1 = max(0, int((y_center - h / 2) * img_size[1]))
            x2 = min(img_size[0] - 1, int((x_center + w / 2) * img_size[0]))
            y2 = min(img_size[1] - 1, int((y_center + h / 2) * img_size[1]))
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, 2)  # 白框，线宽2

    return img

def generate_all_layouts(label_dir, output_dir, img_size=(512, 512)):
    """遍历标签文件夹，生成所有 layout 图像"""
    os.makedirs(output_dir, exist_ok=True)
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    for label_file in tqdm(label_files, desc="Generating layout images"):
        label_path = os.path.join(label_dir, label_file)
        layout_img = draw_yolo_boxes(label_path, img_size)
        save_path = os.path.join(output_dir, label_file.replace(".txt", ".png"))
        cv2.imwrite(save_path, layout_img)

    print(f"✅ Done. Layout images saved to: {output_dir}")


# ============ 修改此处路径为你自己的 ============

LABEL_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/labels"
OUTPUT_LAYOUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/layouts"

generate_all_layouts(LABEL_DIR, OUTPUT_LAYOUT_DIR)
