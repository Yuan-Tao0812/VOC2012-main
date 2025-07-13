import os
import json
from collections import Counter
from tqdm import tqdm

# === 路径配置（改成你自己的实际路径） ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VOC2012/VOC2012-Segmentation-val/labels/"         # YOLO格式标签文件夹
OUTPUT_JSONL_PATH = "/content/drive/MyDrive/VOC2012/VOC2012-Segmentation-val/prompt.json"
SOURCE_IMG_DIR = "segmaps"  # segmentation maps
TARGET_IMG_DIR = "images"  # original images

# VOC 2012 类别（对应语义分割 ID，0是背景，不使用）
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def load_yolo_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    class_ids = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_ids.append(int(parts[0]))
    return class_ids


def generate_prompt_lines(yolo_dir, output_path):
    with open(output_path, 'w') as out_file:
        for label_file in tqdm(os.listdir(yolo_dir)):
            if not label_file.endswith(".txt"):
                continue

            image_id = os.path.splitext(label_file)[0]
            label_path = os.path.join(yolo_dir, label_file)
            class_ids = load_yolo_labels(label_path)

            if not class_ids:
                prompt = "no objects"
            else:
                counts = Counter(class_ids)
                parts = []
                for cls_id, count in sorted(counts.items()):
                    class_name = VOC_CLASSES[cls_id]
                    plural = class_name + "s" if count > 1 else class_name
                    parts.append(f"{count} {plural}")
                prompt = ", ".join(parts)

            line_obj = {
                "source": f"{SOURCE_IMG_DIR}/{image_id}.png",
                "target": f"{TARGET_IMG_DIR}/{image_id}.png",
                "prompt": prompt
            }

            # 一行一个 JSON 对象，无逗号
            out_file.write(json.dumps(line_obj) + "\n")

    print(f"Saved prompt lines to {output_path}")


# === 执行 ===
if __name__ == "__main__":
    generate_prompt_lines(YOLO_LABELS_DIR, OUTPUT_JSONL_PATH)
