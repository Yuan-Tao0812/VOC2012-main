import os
import shutil

# === 你自己手动复制的路径 ===
# 原始路径
VOC_ROOT = "D:/resource/dissertation/VOC2012"
IMG_SRC = os.path.join(VOC_ROOT, "JPEGImages")
LABEL_SRC = os.path.join(VOC_ROOT, "labels")

# 保存路径（你手动设置）
IMG_TRAIN_DST = "D:/resource/dissertation/VOC2012/VOC2012-train/images"
IMG_VAL_DST = "D:/resource/dissertation/VOC2012/VOC2012-val/images"
LABEL_TRAIN_DST = "D:/resource/dissertation/VOC2012/VOC2012-train/labels"
LABEL_VAL_DST = "D:/resource/dissertation/VOC2012/VOC2012-val/labels"

# 读取 train/val 名单
train_list_path = os.path.join(VOC_ROOT, "ImageSets", "Main", "train.txt")
val_list_path = os.path.join(VOC_ROOT, "ImageSets", "Main", "val.txt")

with open(train_list_path, "r") as f:
    train_files = [line.strip() for line in f.readlines()]

with open(val_list_path, "r") as f:
    val_files = [line.strip() for line in f.readlines()]

# 创建目标目录（如果不存在）
os.makedirs(IMG_TRAIN_DST, exist_ok=True)
os.makedirs(IMG_VAL_DST, exist_ok=True)
os.makedirs(LABEL_TRAIN_DST, exist_ok=True)
os.makedirs(LABEL_VAL_DST, exist_ok=True)

def copy_files(file_list, img_dst, label_dst):
    for fname in file_list:
        img_src_path = os.path.join(IMG_SRC, f"{fname}.jpg")
        label_src_path = os.path.join(LABEL_SRC, f"{fname}.txt")

        img_dst_path = os.path.join(img_dst, f"{fname}.jpg")
        label_dst_path = os.path.join(label_dst, f"{fname}.txt")

        if os.path.exists(img_src_path):
            shutil.copyfile(img_src_path, img_dst_path)
        else:
            print(f"Image not found: {img_src_path}")

        if os.path.exists(label_src_path):
            shutil.copyfile(label_src_path, label_dst_path)
        else:
            print(f"Label not found: {label_src_path}")

# 执行复制
copy_files(train_files, IMG_TRAIN_DST, LABEL_TRAIN_DST)
copy_files(val_files, IMG_VAL_DST, LABEL_VAL_DST)

print("✅ 所有图像和标签已按训练/验证划分并复制完成。")
