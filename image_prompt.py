import os
import json
from collections import Counter

# 配置路径
IMAGES_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/images/"
LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/labels/"
OUTPUT_JSONL = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/prompt.jsonl"

# 类别名映射（根据你的 VisDrone 数据集）
CATEGORY_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}


def pluralize(name, count):
    return f"{count} {name + 's' if count > 1 else name}"


def build_prompt(counter):
    parts = [pluralize(CATEGORY_NAMES[c], n) for c, n in sorted(counter.items()) if c in CATEGORY_NAMES]
    if not parts:
        return "There are no objects in the image."
    if len(parts) == 1:
        return f"There is {parts[0]} in the image."
    return f"There are {', '.join(parts[:-1])}, and {parts[-1]} in the image."


def process_dataset():
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg") or f.endswith(".png")]

    with open(OUTPUT_JSONL, "w") as jsonl_file:
        for fname in sorted(image_files):
            image_path = os.path.join(IMAGES_DIR, fname)
            label_path = os.path.join(LABELS_DIR, os.path.splitext(fname)[0] + ".txt")
            layout_name = os.path.splitext(fname)[0] + ".png"  # 语义图名称，假设后续会使用

            counter = Counter()

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls_id = int(parts[0])
                            counter[cls_id] += 1

            prompt = build_prompt(counter)

            json_obj = {
                "image": fname,
                "layout": layout_name,
                "prompt": prompt
            }

            jsonl_file.write(json.dumps(json_obj) + "\n")


if __name__ == "__main__":
    process_dataset()
