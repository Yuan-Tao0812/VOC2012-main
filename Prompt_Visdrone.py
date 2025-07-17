import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

# ========== 配置 ==========
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/labels/"
IMAGES_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/images/"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train-split/"
PROMPT_JSONL = os.path.join(OUTPUT_DIR, "prompt.jsonl")

MAX_OBJECTS = 10
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "layouts"), exist_ok=True)

global_counter = 1  # ✅ 改动1：初始化全局图像编号计数器

def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts)
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append((int(cls_id), x1, y1, x2, y2))
    return boxes

def get_ioa(box, crop_box):
    _, x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop_box
    inter_x1 = max(x1, cx1)
    inter_y1 = max(y1, cy1)
    inter_x2 = min(x2, cx2)
    inter_y2 = min(y2, cy2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    box_area = (x2 - x1) * (y2 - y1)
    return inter_area / box_area if box_area > 0 else 0

def get_boxes_in_crop(boxes, crop_box, threshold=0.5):
    cx1, cy1, cx2, cy2 = crop_box
    selected = []
    for box in boxes:
        cls_id, x1, y1, x2, y2 = box
        ioa = get_ioa(box, crop_box)
        if ioa >= threshold:
            adj_x1 = max(0, x1 - cx1)
            adj_y1 = max(0, y1 - cy1)
            adj_x2 = min(cx2 - cx1, x2 - cx1)
            adj_y2 = min(cy2 - cy1, y2 - cy1)
            selected.append((cls_id, adj_x1, adj_y1, adj_x2, adj_y2))
    return selected

def generate_layout(boxes, width, height):
    layout = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layout)
    for _, x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return layout.convert("RGB")

def save_crop(img, layout, crop_box, boxes, save_name):
    cx1, cy1, cx2, cy2 = map(int, crop_box)
    crop_img = img.crop((cx1, cy1, cx2, cy2))
    crop_layout = generate_layout(boxes, cx2 - cx1, cy2 - cy1)
    crop_layout = crop_layout.crop((0, 0, cx2 - cx1, cy2 - cy1))

    crop_img = crop_img.resize((512, 512), Image.LANCZOS)
    crop_layout = crop_layout.resize((512, 512), Image.LANCZOS)

    img_path = f"images/{save_name}.jpg"
    layout_path = f"layouts/{save_name}.png"

    crop_img.save(os.path.join(OUTPUT_DIR, img_path))
    crop_layout.save(os.path.join(OUTPUT_DIR, layout_path))

    classes = [b[0] for b in boxes]
    desc_parts = []
    for cid in sorted(set(classes)):
        count = classes.count(cid)
        name = VISDRONE_CLASSES[cid]
        desc_parts.append(f"{count} {name}{'s' if count > 1 else ''}")
    left2right = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    sequence = ", ".join([VISDRONE_CLASSES[b[0]] for b in left2right])
    prompt = f"There are {', '.join(desc_parts)} in the image. From left to right: {sequence}."

    return {
        "image_path": img_path,
        "layout_path": layout_path,
        "prompt": prompt
    }

def recursive_crop(img, layout, boxes, crop_box, image_id, prefix):
    cx1, cy1, cx2, cy2 = map(int, crop_box)
    selected_boxes = get_boxes_in_crop(boxes, crop_box, threshold=0.5)

    if len(selected_boxes) <= MAX_OBJECTS:
        global global_counter  # ✅ 改动2
        save_name = f"{global_counter:06d}"
        global_counter += 1
        result = save_crop(img, layout, crop_box, selected_boxes, save_name)
        write_entry(result)  # ✅ 改动3：边裁剪边写入
        show_progress()      # ✅ 改动4：更新数字显示
        return [result]

    width = cx2 - cx1
    height = cy2 - cy1

    results = []
    if width >= height:
        mid_x = (cx1 + cx2) // 2
        results.extend(recursive_crop(img, layout, boxes, (cx1, cy1, mid_x, cy2), image_id, prefix + "_0"))
        results.extend(recursive_crop(img, layout, boxes, (mid_x, cy1, cx2, cy2), image_id, prefix + "_1"))
    else:
        mid_y = (cy1 + cy2) // 2
        results.extend(recursive_crop(img, layout, boxes, (cx1, cy1, cx2, mid_y), image_id, prefix + "_0"))
        results.extend(recursive_crop(img, layout, boxes, (cx1, mid_y, cx2, cy2), image_id, prefix + "_1"))
    return results

written_count = 0  # ✅ 改动5：计数器定义

def write_entry(entry):  # ✅ 改动6：单条写入函数
    global written_count
    with open(PROMPT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    written_count += 1

def show_progress():  # ✅ 改动7：动态进度显示
    print(f"\r已写入数量: {written_count}", end="")

def main():
    if os.path.exists(PROMPT_JSONL):
        os.remove(PROMPT_JSONL)  # ✅ 改动8：清空旧文件

    for label_file in tqdm(os.listdir(YOLO_LABELS_DIR)):
        if not label_file.endswith(".txt"):
            continue

        image_id = label_file[:-4]
        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        label_path = os.path.join(YOLO_LABELS_DIR, label_file)

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path).convert("RGB")
        boxes = load_yolo_boxes(label_path, *img.size)

        if not boxes:
            continue

        width, height = img.size
        full_layout = generate_layout(boxes, width, height)
        crop_box = (0, 0, width, height)
        recursive_crop(img, full_layout, boxes, crop_box, image_id, "0")

    print(f"\n✅ 完成，写入成功，共生成裁剪块数量: {written_count}")  # ✅ 改动9

if __name__ == "__main__":
    main()
