import os
import json
import shutil
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm  # 进度条库

# === 配置路径 ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/labels/"
IMAGES_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/images/"
OUTPUT_ROOT = "/content/drive/MyDrive/VisDrone2019-YOLO/train/"

IMAGES_OUT = os.path.join(OUTPUT_ROOT, "images")
LAYOUTS_OUT = os.path.join(OUTPUT_ROOT, "layouts")
PROMPT_JSONL = os.path.join(OUTPUT_ROOT, "prompt.jsonl")

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
    # 保证坐标在图像范围内
    x1 = max(0, min(x1, img_w-1))
    y1 = max(0, min(y1, img_h-1))
    x2 = max(0, min(x2, img_w-1))
    y2 = max(0, min(y2, img_h-1))
    return int(cls), x1, y1, x2, y2

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
        draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=(0, 0, 0))

def clean_fragments(layout_np, target_mask, color):
    """
    对于目标的颜色区域 target_mask（二值掩码），
    找到所有连通块，保留最大连通块，其它碎块擦黑
    """
    labeled, num = label(target_mask)
    if num <= 1:
        return  # 不用处理

    areas = [(labeled == i).sum() for i in range(1, num+1)]
    max_idx = np.argmax(areas) + 1  # 最大连通域标签

    # 遍历所有碎块，擦黑
    for i in range(1, num+1):
        if i == max_idx:
            continue
        layout_np[labeled == i] = (0, 0, 0)  # 填黑

def main():
    skipped_small_objects = 0  # 新增：统计跳过目标数量

    with open(PROMPT_JSONL, "w") as jsonl_file:
        # 用 tqdm 包装循环，显示进度
        for fname in tqdm(sorted(os.listdir(IMAGES_DIR)), desc="Processing images"):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue

            base = os.path.splitext(fname)[0]
            img_path = os.path.join(IMAGES_DIR, fname)
            label_path = os.path.join(YOLO_LABELS_DIR, base + ".txt")

            if not os.path.exists(label_path):
                print(f"跳过无标签文件: {fname}")
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            layout = Image.new("RGB", (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(layout)
            counter = Counter()

            # 读取所有目标框数据，先存列表，后面分块判断用
            boxes = []
            with open(label_path, "r") as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    if len(vals) != 5:
                        continue
                    cls, x1, y1, x2, y2 = yolo_to_box(vals, w, h)

                    # 新增：跳过太小的目标（无法画边框）
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        skipped_small_objects += 1
                        continue

                    boxes.append((int(cls), (x1, y1, x2, y2)))
                    counter[int(cls)] += 1

            layout_np = np.array(layout)

            # 画每个目标块，带黑框，并即时清理碎块
            for cls, box in boxes:
                color = CATEGORY_COLORS.get(cls, (255, 255, 255))
                x1, y1, x2, y2 = box

                # 画色块
                layout_np[y1:y2+1, x1:x2+1] = color
                layout = Image.fromarray(layout_np)
                draw = ImageDraw.Draw(layout)

                # 画黑框，2像素宽
                add_black_border(draw, box, border=2)

                # 更新 numpy 数组
                layout_np = np.array(layout)

                # 生成当前目标颜色掩码（二值）
                target_mask = np.all(layout_np == color, axis=2)

                # 对碎块做清理，保留最大连通块，擦黑其他碎块
                clean_fragments(layout_np, target_mask, color)

                # 更新 layout 和 draw
                layout = Image.fromarray(layout_np)
                draw = ImageDraw.Draw(layout)

            # 构建prompt
            prompt = build_prompt(counter)

            # 保存layout和复制原图
            layout_name = base + ".png"
            layout.save(os.path.join(LAYOUTS_OUT, layout_name))
            shutil.copy(img_path, os.path.join(IMAGES_OUT, fname))

            # 写jsonl一行
            jsonl_file.write(json.dumps({
                "file_name": fname,
                "layout_file": layout_name,
                "prompt": prompt
            }) + "\n")

    # 统计数据
    num_images = len([f for f in os.listdir(IMAGES_OUT) if f.lower().endswith(('.jpg', '.png'))])
    num_layouts = len([f for f in os.listdir(LAYOUTS_OUT) if f.lower().endswith('.png')])
    with open(PROMPT_JSONL, 'r') as f:
        num_prompts = sum(1 for _ in f)

    print("所有图像处理完成，已生成 layout 和 prompt.jsonl。")
    print(f"图像数量：{num_images}")
    print(f"Layout 数量：{num_layouts}")
    print(f"prompt.jsonl 行数：{num_prompts}")
    print(f"跳过太小的目标：{skipped_small_objects} 个")  # 新增统计输出

if __name__ == "__main__":
    main()
