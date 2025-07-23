import os
import json
from collections import defaultdict


def generate_metadata_jsonl():
    # VOC2012类别名称映射
    class_names = {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }

    # 路径设置
    labels_dir = '/content/drive/MyDrive/VOC2012/VOC2012-train/labels'
    output_dir = '/content/drive/MyDrive/VOC2012/VOC2012-train'
    metadata_file = os.path.join(output_dir, 'metadata.jsonl')

    # 检查标签目录是否存在
    if not os.path.exists(labels_dir):
        print(f"错误: 标签目录不存在: {labels_dir}")
        return 0

    processed_count = 0

    # 获取所有标签文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    with open(metadata_file, 'w', encoding='utf-8') as jsonl_file:
        for label_file in label_files:
            # 获取文件名（不含扩展名）
            filename = os.path.splitext(label_file)[0]

            # 构建路径
            image_path = f"images/{filename}.jpg"
            conditioning_image_path = f"conditioning_images/{filename}.png"

            # 读取标签文件
            label_path = os.path.join(labels_dir, label_file)
            object_counts = defaultdict(int)

            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # YOLO格式: class_id x_center y_center width height
                            parts = line.split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                if class_id in class_names:
                                    object_counts[class_id] += 1

                # 生成文本描述
                if object_counts:
                    # 按类别ID排序
                    sorted_objects = sorted(object_counts.items())
                    descriptions = []

                    for class_id, count in sorted_objects:
                        class_name = class_names[class_id]
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
                        text_description = f"An image of {descriptions[0]}."
                    elif len(descriptions) == 2:
                        text_description = f"An image of {descriptions[0]} and {descriptions[1]}."
                    else:
                        text_description = f"An image of {', '.join(descriptions[:-1])} and {descriptions[-1]}."
                else:
                    # 如果没有检测到对象，使用默认描述
                    text_description = "An image."

                # 创建JSON对象
                metadata_entry = {
                    "image": image_path,
                    "conditioning_image": conditioning_image_path,
                    "text": text_description
                }

                # 写入JSONL文件
                jsonl_file.write(json.dumps(metadata_entry, ensure_ascii=False) + '\n')
                processed_count += 1

            except Exception as e:
                print(f"处理文件 {label_file} 时出错: {e}")
                continue

    print(f"成功生成metadata.jsonl文件: {metadata_file}")
    print(f"总共处理了 {processed_count} 组数据")
    return processed_count


# 运行脚本
if __name__ == "__main__":
    generate_metadata_jsonl()