import os
import json
import shutil
from PIL import Image, ImageDraw
from collections import Counter, defaultdict
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

# === 配置路径 ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VOC2012/VOC2012-train/labels/"
IMAGES_DIR = "/content/drive/MyDrive/VOC2012/VOC2012-train/images/"
LAYOUTS_OUT = "/content/drive/MyDrive/VOC2012/VOC2012-train/conditioning_images"
OUTPUT_DIR = "/content/drive/MyDrive/VOC2012/VOC2012-train"

os.makedirs(LAYOUTS_OUT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORY_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255),
    5: (0, 255, 255), 6: (128, 0, 0), 7: (0, 128, 0), 8: (0, 0, 128), 9: (128, 128, 0),
    10: (255, 128, 0), 11: (128, 255, 0), 12: (0, 255, 128), 13: (128, 0, 255), 14: (255, 128, 128),
    15: (128, 255, 128), 16: (128, 128, 255), 17: (255, 255, 128), 18: (255, 128, 255), 19: (128, 255, 255)
}

CATEGORY_NAMES = {
    0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
    5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
    10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
    15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor"
}


def yolo_to_box(yolo_box, img_w, img_h):
    cls, x, y, w, h = yolo_box
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return int(cls), max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)


def add_black_border(draw, box, border=2):
    x1, y1, x2, y2 = box
    for i in range(border):
        draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=(0, 0, 0))


def clean_target_fragments(layout_np, mask_map, target_id, box, color):
    x1, y1, x2, y2 = box
    region_mask = (mask_map[y1:y2 + 1, x1:x2 + 1] == target_id)
    color_mask = np.all(layout_np[y1:y2 + 1, x1:x2 + 1] == color, axis=2)
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
        layout_np[y1:y2 + 1, x1:x2 + 1][mask] = (0, 0, 0)


def generate_text_description(object_counts):
    """根据对象计数生成文本描述"""
    if not object_counts:
        return "An image."

    # 按类别ID排序
    sorted_objects = sorted(object_counts.items())
    descriptions = []

    for class_id, count in sorted_objects:
        class_name = CATEGORY_NAMES[class_id]
        # 处理复数形式
        if count == 1:
            if class_name.endswith('s') or class_name in ['sheep', 'person']:
                if class_name == 'person':
                    descriptions.append(f"{count} person")
                else:
                    descriptions.append(f"{count} {class_name}")
            else:
                descriptions.append(f"{count} {class_name}")
        else:
            if class_name == 'person':
                descriptions.append(f"{count} persons")
            elif class_name.endswith('s'):
                descriptions.append(f"{count} {class_name}")
            elif class_name == 'sheep':
                descriptions.append(f"{count} sheep")
            else:
                descriptions.append(f"{count} {class_name}s")

    if len(descriptions) == 1:
        return f"An image of {descriptions[0]}."
    elif len(descriptions) == 2:
        return f"An image of {descriptions[0]} and {descriptions[1]}."
    else:
        return f"An image of {', '.join(descriptions[:-1])} and {descriptions[-1]}."


def main():
    skipped_small_objects = 0
    images_with_small_objects = []  # 记录包含小目标的图像编号
    metadata_file = os.path.join(OUTPUT_DIR, 'metadata.jsonl')

    # 获取所有图像文件并排序
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
    image_files.sort()

    with open(metadata_file, 'w', encoding='utf-8') as jsonl_file:
        for fname in tqdm(image_files, desc="Processing images"):
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
            object_counts = defaultdict(int)  # 用于生成文本描述的对象计数
            target_id = 0
            has_small_objects = False  # 标记当前图像是否包含小目标

            with open(label_path, "r") as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    if len(vals) != 5:
                        continue
                    cls, x1, y1, x2, y2 = yolo_to_box(vals, w, h)

                    # 检查目标大小，过滤过小的目标
                    if x2 - x1 < 6 or y2 - y1 < 6:
                        skipped_small_objects += 1
                        has_small_objects = True
                        continue

                    # 只有通过大小检查的目标才会被添加到boxes和计数中
                    boxes.append((target_id, cls, (x1, y1, x2, y2)))
                    counter[cls] += 1
                    object_counts[cls] += 1  # 用于文本描述的计数
                    target_id += 1

            # 如果当前图像包含小目标，记录其编号
            if has_small_objects:
                images_with_small_objects.append(base)

            # 生成布局图像
            layout_np = np.array(layout)
            for tid, cls, box in boxes:
                x1, y1, x2, y2 = box
                color = CATEGORY_COLORS.get(cls, (255, 255, 255))
                layout_np[y1:y2 + 1, x1:x2 + 1] = color
                mask_map[y1:y2 + 1, x1:x2 + 1] = tid
                layout = Image.fromarray(layout_np)
                draw = ImageDraw.Draw(layout)
                add_black_border(draw, box, border=2)
                layout_np = np.array(layout)
                mask_map[y1:y2 + 1, x1:x2 + 1][np.all(layout_np[y1:y2 + 1, x1:x2 + 1] == (0, 0, 0), axis=2)] = -1

            # 清理碎片
            for tid, cls, box in boxes:
                color = CATEGORY_COLORS.get(cls, (255, 255, 255))
                clean_target_fragments(layout_np, mask_map, tid, box, color)

            # 保存布局图像
            layout = Image.fromarray(layout_np)
            layout_name = base + ".png"
            layout.save(os.path.join(LAYOUTS_OUT, layout_name))

            # 生成文本描述（只包含未被过滤的目标）
            text_description = generate_text_description(object_counts)

            # 创建metadata条目
            metadata_entry = {
                "image": f"images/{fname}",
                "conditioning_image": f"conditioning_images/{layout_name}",
                "text": text_description
            }

            # 写入JSONL文件
            jsonl_file.write(json.dumps(metadata_entry, ensure_ascii=False) + '\n')

    print("\n所有图像处理完成。")
    print(f"跳过太小的目标：{skipped_small_objects} 个")
    print(f"包含小目标的图像数量：{len(images_with_small_objects)} 个")
    if images_with_small_objects:
        print(f"包含小目标的图像编号：{', '.join(images_with_small_objects)}")
    print(f"成功生成metadata.jsonl文件: {metadata_file}")

    # 验证生成的文件
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            actual_lines = sum(1 for _ in f)
        print(f"metadata.jsonl实际包含 {actual_lines} 行")


if __name__ == "__main__":
    main()