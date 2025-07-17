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
    objs = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            objs.append((x_center, class_id))
    return objs

def generate_prompt_lines(yolo_dir, output_path):
    with open(output_path, 'w') as out_file:
        for label_file in tqdm(os.listdir(yolo_dir)):
            if not label_file.endswith(".txt"):
                continue

            image_id = os.path.splitext(label_file)[0]
            label_path = os.path.join(yolo_dir, label_file)
            objs = load_yolo_labels(label_path)

            if not objs:
                prompt = "no objects, visdrone style"
            else:
                class_ids = [cid for _, cid in objs]
                counts = Counter(class_ids)
                summary_parts = []
                for cls_id, count in sorted(counts.items()):
                    name = VISDRONE_CLASSES[cls_id]
                    plural = name + "s" if count > 1 else name
                    summary_parts.append(f"{count} {plural}")
                summary_str = "There are " + " and ".join(summary_parts) + " in the image."

                sorted_objs = sorted(objs, key=lambda x: x[0])  # 按 x_center 排序
                sequence = ", ".join(VISDRONE_CLASSES[cid] for _, cid in sorted_objs)
                prompt = f"{summary_str} From left to right: {sequence}. visdrone style"

            line_obj = {
                "image_path": f"{IMAGES_RELATIVE_DIR}/{image_id}.jpg",
                "layout_path": f"{LAYOUTS_RELATIVE_DIR}/{image_id}.png",
                "prompt": prompt
            }
            out_file.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

    print(f"✅ Saved prompt lines to {output_path}")

if __name__ == "__main__":
    generate_prompt_lines(YOLO_LABELS_DIR, OUTPUT_JSONL_PATH)
