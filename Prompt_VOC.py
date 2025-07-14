import os
import json
from collections import Counter
from tqdm import tqdm

# === 路径配置 ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VOC2012/VOC2012-train/labels/"
OUTPUT_JSONL_PATH = "/content/drive/MyDrive/VOC2012/VOC2012-train/prompt.json"
IMAGES_RELATIVE_DIR = "images"  # 这里是相对路径，训练脚本会用这个路径找到图片

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
                prompt = "no objects, voc2012 style"
            else:
                counts = Counter(class_ids)
                parts = []
                for cls_id, count in sorted(counts.items()):
                    class_name = VOC_CLASSES[cls_id]
                    plural = class_name + "s" if count > 1 else class_name
                    parts.append(f"{count} {plural}")
                prompt = ", ".join(parts) + ", voc2012 style"

            line_obj = {
                "image_path": f"{IMAGES_RELATIVE_DIR}/{image_id}.jpg",
                "prompt": prompt
            }
            # 一行写入一个完整的json对象，末尾不加逗号
            out_file.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

    print(f"Saved prompt lines to {output_path}")


if __name__ == "__main__":
    generate_prompt_lines(YOLO_LABELS_DIR, OUTPUT_JSONL_PATH)
