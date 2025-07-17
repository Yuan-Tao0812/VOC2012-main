import os
import shutil
from PIL import Image, ImageDraw

# 原始路径
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/labels/"
IMAGES_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/images/"

# 新保存路径
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/"
IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "images")
LBL_OUT_DIR = os.path.join(OUTPUT_DIR, "labels")
VIS_OUT_DIR = os.path.join(OUTPUT_DIR, "vis")

# 创建目录
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(LBL_OUT_DIR, exist_ok=True)
os.makedirs(VIS_OUT_DIR, exist_ok=True)

# 获取图像文件并排序（防止乱序）
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])

# 处理并重命名
count = 1
for img_file in image_files:
    image_id = os.path.splitext(img_file)[0]
    label_file = image_id + ".txt"

    img_path = os.path.join(IMAGES_DIR, img_file)
    lbl_path = os.path.join(YOLO_LABELS_DIR, label_file)

    if not os.path.exists(lbl_path):
        continue  # 如果没有标签，跳过

    # 复制图像和标签到新位置并重命名
    new_name = f"{count}"
    shutil.copy(img_path, os.path.join(IMG_OUT_DIR, new_name + ".jpg"))
    shutil.copy(lbl_path, os.path.join(LBL_OUT_DIR, new_name + ".txt"))

    # 可视化标签（画框）
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = map(float, parts)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    img.save(os.path.join(VIS_OUT_DIR, new_name + ".jpg"))

    count += 1

print(f"✅ 总图像数量：{count - 1}")
print(f"📁 图像保存于：{IMG_OUT_DIR}")
print(f"📁 标签保存于：{LBL_OUT_DIR}")
print(f"📁 可视化图像保存于：{VIS_OUT_DIR}")
