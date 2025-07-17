import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import Counter
import numpy as np

# === 配置路径 ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/labels/"
IMAGES_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/images/"
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

def generate_layout_image(boxes, img_w, img_h):
    layout = Image.new("L", (img_w, img_h), 0)  # 单通道黑底
    draw = ImageDraw.Draw(layout)
    for cls_id, x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=255, width=2)  # 白色轮廓线，宽度2
    return layout.convert("RGB")  # 如果后续需要3通道图，转回RGB

def split_into_dynamic_crops(boxes, img_w, img_h, max_objs=MAX_OBJECTS_PER_CROP, grid_size=4):
    grid_w, grid_h = img_w / grid_size, img_h / grid_size
    grid_counts = np.zeros((grid_size, grid_size), dtype=int)
    grid_boxes = [[[] for _ in range(grid_size)] for _ in range(grid_size)]

    for box in boxes:
        _, x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        gx = min(int(cx // grid_w), grid_size-1)
        gy = min(int(cy // grid_h), grid_size-1)
        grid_counts[gy, gx] += 1
        grid_boxes[gy][gx].append(box)

    crops = []
    visited = np.zeros_like(grid_counts, dtype=bool)

    for gy in range(grid_size):
        for gx in range(grid_size):
            if visited[gy, gx]:
                continue
            count = grid_counts[gy, gx]
            if count == 0:
                visited[gy, gx] = True
                continue

            if count >= max_objs:
                crops.append(grid_boxes[gy][gx])
                visited[gy, gx] = True
            else:
                merged_boxes = list(grid_boxes[gy][gx])
                visited[gy, gx] = True

                # 简单向右和向下合并
                for dy, dx in [(0,1), (1,0)]:
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size and not visited[ny, nx]:
                        if grid_counts[ny, nx] > 0 and len(merged_boxes) + grid_counts[ny, nx] <= max_objs:
                            merged_boxes.extend(grid_boxes[ny][nx])
                            visited[ny, nx] = True
                crops.append(merged_boxes)
    return crops

def get_crop_bbox(boxes, pad, img_w, img_h):
    x1 = min(b[1] for b in boxes)
    y1 = min(b[2] for b in boxes)
    x2 = max(b[3] for b in boxes)
    y2 = max(b[4] for b in boxes)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + pad)

    return int(x1), int(y1), int(x2), int(y2)

def crop_and_save(img, layout, boxes, crop_idx, image_id):
    img_w, img_h = img.size
    pad = 20

    x1, y1, x2, y2 = get_crop_bbox(boxes, pad, img_w, img_h)

    crop_img = img.crop((x1,y1,x2,y2)).resize((IMAGE_SIZE, IMAGE_SIZE))
    crop_layout = layout.crop((x1,y1,x2,y2)).resize((IMAGE_SIZE, IMAGE_SIZE))

    crop_objects = [b[0] for b in boxes]
    rel_boxes = [(cls_id, (b[1]+b[3])/2 - x1) for cls_id, *b in boxes]

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

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        raw_boxes = load_yolo_annotations(label_path)
        abs_boxes = [yolo_to_absolute(box, w, h) for box in raw_boxes]

        if not abs_boxes:
            continue

        # 根据标签生成layout图
        layout = generate_layout_image(abs_boxes, w, h)

        # 根据目标密度动态裁剪成不同大小的块
        crops = split_into_dynamic_crops(abs_boxes, w, h)

        for i, crop_boxes in enumerate(crops):
            entry = crop_and_save(img, layout, crop_boxes, i, image_id)
            all_entries.append(entry)

    with open(PROMPT_JSONL, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Done. {len(all_entries)} crops saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
