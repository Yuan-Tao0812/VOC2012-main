import os
import json
import shutil
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter

# === 配置路径 ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/labels/"
IMAGES_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/images/"
LAYOUTS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/layouts/"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train-split/"
PROMPT_JSONL = os.path.join(OUTPUT_DIR, "prompt.jsonl")

MAX_OBJECTS_PER_CROP = 10
IMAGE_SIZE = 512

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "layouts"), exist_ok=True)

def load_yolo_annotations(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts)
            boxes.append((int(cls_id), cx, cy, w, h))
    return boxes

def yolo_to_absolute(box, img_w, img_h):
    cls_id, cx, cy, w, h = box
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return cls_id, max(0,x1), max(0,y1), min(img_w,x2), min(img_h,y2)

def crop_image_and_layout(img, layout, crop_boxes, crop_idx, image_id):
    crop_prompts = []
    crop_objects = []

    x1 = min(b[1] for b in crop_boxes)
    y1 = min(b[2] for b in crop_boxes)
    x2 = max(b[3] for b in crop_boxes)
    y2 = max(b[4] for b in crop_boxes)

    pad = 20
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, img.width)
    y2 = min(y2 + pad, img.height)

    crop_img = img.crop((x1, y1, x2, y2)).resize((IMAGE_SIZE, IMAGE_SIZE))
    crop_layout = layout.crop((x1, y1, x2, y2)).resize((IMAGE_SIZE, IMAGE_SIZE))

    rel_boxes = []
    for box in crop_boxes:
        cls_id, bx1, by1, bx2, by2 = box
        rel_boxes.append((cls_id, (bx1 + bx2)/2 - x1))  # 用于从左到右排序
        crop_objects.append(cls_id)

    counter = Counter(crop_objects)
    desc = [f"{count} {VISDRONE_CLASSES[cls]}{'s' if count > 1 else ''}" for cls, count in sorted(counter.items())]
    cls_order = sorted(rel_boxes, key=lambda x: x[1])
    sequence = ", ".join([VISDRONE_CLASSES[cls] for cls, _ in cls_order])

    prompt = f"There are {', '.join(desc)} in the image. From left to right: {sequence}."

    crop_name = f"{image_id}_crop{crop_idx}"
    crop_img.save(os.path.join(OUTPUT_DIR, "images", f"{crop_name}.jpg"))
    crop_layout.save(os.path.join(OUTPUT_DIR, "layouts", f"{crop_name}.png"))

    return {
        "image_path": f"images/{crop_name}.jpg",
        "layout_path": f"layouts/{crop_name}.png",
        "prompt": prompt
    }

def main():
    all_entries = []
    label_files = [f for f in os.listdir(YOLO_LABELS_DIR) if f.endswith(".txt")]

    for label_file in tqdm(label_files):
        image_id = label_file.replace(".txt", "")
        label_path = os.path.join(YOLO_LABELS_DIR, label_file)
        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        layout_path = os.path.join(LAYOUTS_DIR, f"{image_id}.png")

        if not os.path.exists(image_path) or not os.path.exists(layout_path):
            continue

        img = Image.open(image_path).convert("RGB")
        layout = Image.open(layout_path).convert("RGB")
        w, h = img.size

        raw_boxes = load_yolo_annotations(label_path)
        abs_boxes = [yolo_to_absolute(box, w, h) for box in raw_boxes]

        if not abs_boxes:
            continue

        # 分块，每块最多 MAX_OBJECTS_PER_CROP 个目标
        for i in range(0, len(abs_boxes), MAX_OBJECTS_PER_CROP):
            crop_boxes = abs_boxes[i:i+MAX_OBJECTS_PER_CROP]
            entry = crop_image_and_layout(img, layout, crop_boxes, i // MAX_OBJECTS_PER_CROP, image_id)
            all_entries.append(entry)

    with open(PROMPT_JSONL, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Done. {len(all_entries)} crops saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
