import os
import json
import shutil
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

# === 配置路径 ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/labels/"
IMAGES_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/images/"
OUTPUT_ROOT = "/content/drive/MyDrive/VisDrone2019-YOLO/train/"

IMAGES_OUT = os.path.join(OUTPUT_ROOT, "images")
LAYOUTS_OUT = os.path.join(OUTPUT_ROOT, "layouts")
PROMPT_JSONL = os.path.join(OUTPUT_ROOT, "prompt.jsonl")

os.makedirs(IMAGES_OUT, exist_ok=True)
os.makedirs(LAYOUTS_OUT, exist_ok=True)

CATEGORY_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 0),
    7: (0, 128, 0),
    8: (0, 0, 128),
    9: (128, 128, 0),
}

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

def yolo_to_box(yolo_box, img_w, img_h):
    cls, x, y, w, h = yolo_box
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return int(cls), max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)

def pluralize(word, count):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"

def build_prompt(counter):
    parts = [pluralize(CATEGORY_NAMES[c], n) for c, n in sorted(counter.items())]
    if not parts:
        return "There are no objects in the image."
    if len(parts) == 1:
        return f"There is {parts[0]} in the image."
    return f"There are {', '.join(parts[:-1])}, and {parts[-1]} in the image."

def add_black_border(draw, box, border=2):
    x1, y1, x2, y2 = box
    for i in range(border):
        draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=(0, 0, 0))

def clean_single_object_fragments(layout_np, box, color):
    x1, y1, x2, y2 = box
    region = layout_np[y1:y2+1, x1:x2+1]
    mask = np.all(region == color, axis=2)
    labeled, num = label(mask)
    if num <= 1:
        return  # no fragments or only one
    areas = [(labeled == i).sum() for i in range(1, num+1)]
    max_idx = np.argmax(areas) + 1
    for i in range(1, num+1):
        if i != max_idx:
            region[labeled == i] = (0, 0, 0)
    layout_np[y1:y2+1, x1:x2+1] = region

def main():
    skipped_small_objects = 0
    with open(PROMPT_JSONL, "w") as jsonl_file:
        for fname in tqdm(sorted(os.listdir(IMAGES_DIR)), desc="Processing images"):
            if not fname.lower().endswith((".jpg", ".png")):
                continue

            base = os.path.splitext(fname)[0]
            img_path = os.path.join(IMAGES_DIR, fname)
            label_path = os.path.join(YOLO_LABELS_DIR, base + ".txt")
            if not os.path.exists(label_path):
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            layout = Image.new("RGB", (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(layout)
            counter = Counter()
            boxes = []

            with open(label_path, "r") as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    if len(vals) != 5:
                        continue
                    cls, x1, y1, x2, y2 = yolo_to_box(vals, w, h)
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        skipped_small_objects += 1
                        continue
                    boxes.append((cls, (x1, y1, x2, y2)))
                    counter[cls] += 1

            layout_np = np.array(layout)

            for cls, box in boxes:
                color = CATEGORY_COLORS[cls]
                x1, y1, x2, y2 = box
                layout_np[y1:y2+1, x1:x2+1] = color
                layout = Image.fromarray(layout_np)
                draw = ImageDraw.Draw(layout)
                add_black_border(draw, box, border=2)
                layout_np = np.array(layout)

            for cls, box in boxes:
                color = CATEGORY_COLORS[cls]
                clean_single_object_fragments(layout_np, box, color)

            layout = Image.fromarray(layout_np)
            layout_name = base + ".png"
            layout.save(os.path.join(LAYOUTS_OUT, layout_name))
            shutil.copy(img_path, os.path.join(IMAGES_OUT, fname))

            prompt = build_prompt(counter)
            jsonl_file.write(json.dumps({
                "file_name": fname,
                "layout_file": layout_name,
                "prompt": prompt
            }) + "\n")

    print("\nAll images processed.")
    print(f"Total skipped small objects: {skipped_small_objects}")

if __name__ == "__main__":
    main()
