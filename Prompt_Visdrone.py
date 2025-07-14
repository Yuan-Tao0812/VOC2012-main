import os
import json
from collections import Counter
from tqdm import tqdm

# === 路径配置 ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/labels/"  # YOLO格式标签文件夹
OUTPUT_JSONL_PATH = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/prompt.jsonl"
IMAGES_RELATIVE_DIR = "images"    # 图片相对路径
LAYOUTS_RELATIVE_DIR = "layouts"  # layout图相对路径

# VisDrone 类别
VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
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
                prompt = "no objects, visdrone style"
            else:
                counts = Counter(class_ids)
                parts = []
                for cls_id, count in sorted(counts.items()):
                    class_name = VISDRONE_CLASSES[cls_id]
                    plural = class_name + "s" if count > 1 else class_name
                    parts.append(f"{count} {plural}")
                prompt = ", ".join(parts) + ", visdrone style"

            line_obj = {
                "image_path": f"{IMAGES_RELATIVE_DIR}/{image_id}.jpg",
                "layout_path": f"{LAYOUTS_RELATIVE_DIR}/{image_id}.png",
                "prompt": prompt
            }
            out_file.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

    print(f"✅ Saved prompt lines with layout_path to {output_path}")

if __name__ == "__main__":
    generate_prompt_lines(YOLO_LABELS_DIR, OUTPUT_JSONL_PATH)
