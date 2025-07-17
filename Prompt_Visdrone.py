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
    """
    计算目标框在裁剪框内的交叠比例（面积比）
    """
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
    """
    获取所有在crop_box中面积比例≥threshold的目标框，坐标转换为相对crop_box的坐标
    """
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
    """
    生成布局图，只有目标对应区域画白色，背景黑色
    """
    layout = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layout)
    for _, x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return layout.convert("RGB")

def save_crop(img, layout, crop_box, boxes, save_name):
    """
    裁剪原图和布局图，保存到磁盘
    """
    cx1, cy1, cx2, cy2 = map(int, crop_box)
    crop_img = img.crop((cx1, cy1, cx2, cy2))
    crop_layout = generate_layout(boxes, cx2 - cx1, cy2 - cy1)
    crop_layout = crop_layout.crop((0, 0, cx2 - cx1, cy2 - cy1))

    # 缩放到512x512
    crop_img = crop_img.resize((512, 512), Image.LANCZOS)
    crop_layout = crop_layout.resize((512, 512), Image.LANCZOS)

    img_path = f"images/{save_name}.jpg"
    layout_path = f"layouts/{save_name}.png"

    crop_img.save(os.path.join(OUTPUT_DIR, img_path))
    crop_layout.save(os.path.join(OUTPUT_DIR, layout_path))

    # 生成prompt
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
    """
    递归裁剪主函数：
    - 仅保留面积≥50%的目标
    - 目标数≤10则保存
    - 否则沿长边二分递归细分
    - 无最小尺寸限制
    """
    cx1, cy1, cx2, cy2 = map(int, crop_box)
    selected_boxes = get_boxes_in_crop(boxes, crop_box, threshold=0.5)

    if len(selected_boxes) <= MAX_OBJECTS:
        save_name = f"{image_id}_{prefix}"
        return [save_crop(img, layout, crop_box, selected_boxes, save_name)]

    width = cx2 - cx1
    height = cy2 - cy1

    results = []
    if width >= height:
        mid_x = (cx1 + cx2) // 2
        left_crop = (cx1, cy1, mid_x, cy2)
        right_crop = (mid_x, cy1, cx2, cy2)
        results.extend(recursive_crop(img, layout, boxes, left_crop, image_id, prefix + "_0"))
        results.extend(recursive_crop(img, layout, boxes, right_crop, image_id, prefix + "_1"))
    else:
        mid_y = (cy1 + cy2) // 2
        top_crop = (cx1, cy1, cx2, mid_y)
        bottom_crop = (cx1, mid_y, cx2, cy2)
        results.extend(recursive_crop(img, layout, boxes, top_crop, image_id, prefix + "_0"))
        results.extend(recursive_crop(img, layout, boxes, bottom_crop, image_id, prefix + "_1"))

    return results

def main():
    all_data = []
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

        # 初始裁剪区域是整张图
        crop_box = (0, 0, width, height)

        results = recursive_crop(img, full_layout, boxes, crop_box, image_id, "0")
        all_data.extend(results)

    with open(PROMPT_JSONL, "w", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 完成，生成裁剪块数量: {len(all_data)}")

if __name__ == "__main__":
    main()
