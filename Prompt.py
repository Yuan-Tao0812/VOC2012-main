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
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255),
    5: (0, 255, 255), 6: (128, 0, 0), 7: (0, 128, 0), 8: (0, 0, 128), 9: (128, 128, 0)
}
CATEGORY_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
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

def add_black_border(draw, box, border=4):
    x1, y1, x2, y2 = box
    for i in range(border):
        draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=(0, 0, 0))

def clean_target_fragments(layout_np, mask_map, target_id, box, color):
    x1, y1, x2, y2 = box
    region_mask = (mask_map[y1:y2+1, x1:x2+1] == target_id)
    color_mask = np.all(layout_np[y1:y2+1, x1:x2+1] == color, axis=2)
    target_mask = np.logical_and(region_mask, color_mask)

    labeled, num = label(target_mask)
    if num <= 1:
        return
    areas = [(labeled == i).sum() for i in range(1, num + 1)]
    max_idx = np.argmax(areas) + 1
    for i in range(1, num + 1):
        if i == max_idx:
            continue
        mask = (labeled == i)
        layout_np[y1:y2+1, x1:x2+1][mask] = (0, 0, 0)

def main():
    skipped_small_objects = 0
    with open(PROMPT_JSONL, "w") as jsonl_file:
        for fname in tqdm(sorted(os.listdir(IMAGES_DIR)), desc="Processing images"):
            if not fname.lower().endswith(('.jpg', '.png')):
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
            mask_map = np.full((h, w), fill_value=-2, dtype=np.int32)  # -2 = 未处理

            boxes = []
            counter = Counter()
            target_id = 0
            with open(label_path, "r") as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    if len(vals) != 5:
                        continue
                    cls, x1, y1, x2, y2 = yolo_to_box(vals, w, h)
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        skipped_small_objects += 1
                        continue
                    boxes.append((target_id, cls, (x1, y1, x2, y2)))
                    counter[cls] += 1
                    target_id += 1

            layout_np = np.array(layout)
            for tid, cls, box in boxes:
                x1, y1, x2, y2 = box
                color = CATEGORY_COLORS.get(cls, (255, 255, 255))
                layout_np[y1:y2+1, x1:x2+1] = color
                mask_map[y1:y2+1, x1:x2+1] = tid
                layout = Image.fromarray(layout_np)
                draw = ImageDraw.Draw(layout)
                add_black_border(draw, box, border=4)
                layout_np = np.array(layout)
                mask_map[y1:y2+1, x1:x2+1][np.all(layout_np[y1:y2+1, x1:x2+1] == (0, 0, 0), axis=2)] = -1

            # 清理碎片
            for tid, cls, box in boxes:
                color = CATEGORY_COLORS.get(cls, (255, 255, 255))
                clean_target_fragments(layout_np, mask_map, tid, box, color)

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

    print("\n所有图像处理完成。")
    print(f"跳过太小的目标：{skipped_small_objects} 个")

if __name__ == "__main__":
    main()
