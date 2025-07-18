import os
import json
import shutil
from PIL import Image, ImageDraw
from collections import Counter

# === 配置路径 ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/labels/"
IMAGES_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/images/"
OUTPUT_ROOT = "/content/drive/MyDrive/VisDrone2019-YOLO/train/"

IMAGES_OUT = os.path.join(OUTPUT_ROOT, "images")
LAYOUTS_OUT = os.path.join(OUTPUT_ROOT, "layouts")
PROMPT_JSONL = os.path.join(OUTPUT_ROOT, "prompt.jsonl")

# 创建文件夹
os.makedirs(IMAGES_OUT, exist_ok=True)
os.makedirs(LAYOUTS_OUT, exist_ok=True)

# 类别颜色和名称
CATEGORY_COLORS = {
    0: (255, 0, 0),      # pedestrian - 红
    1: (0, 255, 0),      # people - 绿
    2: (0, 0, 255),      # bicycle - 蓝
    3: (255, 255, 0),    # car - 黄
    4: (255, 0, 255),    # van - 紫
    5: (0, 255, 255),    # truck - 青
    6: (128, 0, 0),      # tricycle - 深红
    7: (0, 128, 0),      # awning-tricycle - 深绿
    8: (0, 0, 128),      # bus - 深蓝
    9: (128, 128, 0),    # motor - 橄榄
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
    x1 = int((x - w/2) * img_w)
    y1 = int((y - h/2) * img_h)
    x2 = int((x + w/2) * img_w)
    y2 = int((y + h/2) * img_h)
    return cls, x1, y1, x2, y2

def pluralize(word, count):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"

def build_prompt(counter):
    parts = [pluralize(CATEGORY_NAMES[c], n) for c, n in counter.items()]
    if not parts:
        return "There are no objects in the image."
    if len(parts) == 1:
        return f"There is {parts[0]} in the image."
    return f"There are {', '.join(parts[:-1])}, and {parts[-1]} in the image."

# 写入 JSONL
with open(PROMPT_JSONL, "w") as jsonl_file:
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if not fname.endswith(('.jpg', '.png')):
            continue

        base = os.path.splitext(fname)[0]
        img_path = os.path.join(IMAGES_DIR, fname)
        label_path = os.path.join(YOLO_LABELS_DIR, base + ".txt")

        if not os.path.exists(label_path):
            continue

        # 加载图像
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 创建 layout
        layout = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(layout)
        counter = Counter()

        # 读取标签并画图
        with open(label_path, "r") as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) != 5:
                    continue
                cls, x1, y1, x2, y2 = yolo_to_box(vals, w, h)
                color = CATEGORY_COLORS.get(int(cls), (255, 255, 255))
                draw.rectangle([x1, y1, x2, y2], fill=color)
                counter[int(cls)] += 1

        # 构建prompt
        prompt = build_prompt(counter)

        # 保存 layout 和复制原图
        layout_name = base + "_layout.png"
        layout.save(os.path.join(LAYOUTS_OUT, layout_name))
        shutil.copy(img_path, os.path.join(IMAGES_OUT, fname))

        # 写一行 JSONL
        jsonl_file.write(json.dumps({
            "file_name": fname,
            "layout_file": layout_name,
            "prompt": prompt
        }) + "\n")

print("✅ 所有图像处理完成，已生成 layout 和 prompt.jsonl。")
