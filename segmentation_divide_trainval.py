import os
import shutil

# === 原始数据路径（你只需改这些） ===
VOC_ROOT = r"D:/resource/dissertation/VOC2012"
IMG_DIR = os.path.join(VOC_ROOT, "JPEGImages")
LABEL_DIR = os.path.join(VOC_ROOT, "labels")
SEGMAP_DIR = os.path.join(VOC_ROOT, "SegmentationClass")

TRAIN_TXT = os.path.join(VOC_ROOT, "ImageSets/Segmentation/train.txt")
VAL_TXT = os.path.join(VOC_ROOT, "ImageSets/Segmentation/val.txt")

# === 输出路径（你复制路径改这几个即可） ===
OUTPUT_TRAIN_IMG = r"D:/resource/dissertation/VOC2012/VOC2012-Segmentation-train/images"
OUTPUT_TRAIN_LABEL = r"D:/resource/dissertation/VOC2012/VOC2012-Segmentation-train/labels"
OUTPUT_TRAIN_SEGMAP = r"D:/resource/dissertation/VOC2012/VOC2012-Segmentation-train/segmaps"

OUTPUT_VAL_IMG = r"D:/resource/dissertation/VOC2012/VOC2012-Segmentation-val/images"
OUTPUT_VAL_LABEL = r"D:/resource/dissertation/VOC2012/VOC2012-Segmentation-val/labels"
OUTPUT_VAL_SEGMAP = r"D:/resource/dissertation/VOC2012/VOC2012-Segmentation-val/segmaps"

# === 自动创建输出目录 ===
for d in [OUTPUT_TRAIN_IMG, OUTPUT_TRAIN_LABEL, OUTPUT_TRAIN_SEGMAP,
          OUTPUT_VAL_IMG, OUTPUT_VAL_LABEL, OUTPUT_VAL_SEGMAP]:
    os.makedirs(d, exist_ok=True)

# === 拷贝函数 ===
def copy_split(txt_path, img_out, label_out, segmap_out):
    with open(txt_path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    for image_id in ids:
        img_src = os.path.join(IMG_DIR, image_id + ".jpg")
        label_src = os.path.join(LABEL_DIR, image_id + ".txt")
        segmap_src = os.path.join(SEGMAP_DIR, image_id + ".png")

        shutil.copyfile(img_src, os.path.join(img_out, image_id + ".jpg"))
        if os.path.exists(label_src):
            shutil.copyfile(label_src, os.path.join(label_out, image_id + ".txt"))
        if os.path.exists(segmap_src):
            shutil.copyfile(segmap_src, os.path.join(segmap_out, image_id + ".png"))

# === 执行拷贝 ===
copy_split(TRAIN_TXT, OUTPUT_TRAIN_IMG, OUTPUT_TRAIN_LABEL, OUTPUT_TRAIN_SEGMAP)
copy_split(VAL_TXT, OUTPUT_VAL_IMG, OUTPUT_VAL_LABEL, OUTPUT_VAL_SEGMAP)

print("✅ 所有文件已成功复制到新路径。")
