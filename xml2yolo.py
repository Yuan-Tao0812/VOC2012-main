import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml

# === 直接用字符串指定路径 ===
VOC_ROOT = "D:/resource/dissertation/VOC2012"  # 改成你实际解压的 VOC2012 路径
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")
YOLO_LABELS_DIR = os.path.join(VOC_ROOT, "labels")  # 输出的 YOLO 标签目录
VOC_YAML = "D:/resource/dissertation/VOC2012-main/VOC.yaml"  # 你的 voc.yaml 路径

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# === 加载类别名称 ===
with open(VOC_YAML, "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)
names = list(yaml_data["names"].values())

# === 转换函数 ===
def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_label(xml_path, output_txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(output_txt_path, "w") as out_file:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in names:
                continue
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue
            xmlbox = obj.find("bndbox")
            coords = [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")]
            bb = convert_box((w, h), coords)
            cls_id = names.index(cls)
            out_file.write(" ".join([str(cls_id)] + [f"{a:.6f}" for a in bb]) + "\n")

# === 批量处理 ===
xml_files = os.listdir(ANNOTATIONS_DIR)
for xml_file in tqdm(xml_files, desc="Converting VOC to YOLO"):
    if not xml_file.endswith(".xml"):
        continue
    image_id = os.path.splitext(xml_file)[0]
    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    output_path = os.path.join(YOLO_LABELS_DIR, f"{image_id}.txt")
    convert_label(xml_path, output_path)
