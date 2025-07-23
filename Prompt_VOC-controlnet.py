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
    error_count = 0

    # 获取所有标签文件并按数字顺序排序
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    # 按文件名中的数字排序
    def get_file_number(filename):
        # 提取文件名中的数字部分
        name_without_ext = os.path.splitext(filename)[0]
        try:
            return int(name_without_ext)
        except ValueError:
            return float('inf')  # 如果不是纯数字，放到最后

    label_files.sort(key=get_file_number)

    print(f"找到 {len(label_files)} 个标签文件")
    print(f"第一个文件: {label_files[0] if label_files else '无'}")
    print(f"最后一个文件: {label_files[-1] if label_files else '无'}")

    if len(label_files) == 0:
        print("警告: 没有找到任何.txt标签文件")
        return 0

    with open(metadata_file, 'w', encoding='utf-8') as jsonl_file:
        for i, label_file in enumerate(label_files):
            # 显示进度
            if (i + 1) % 1000 == 0 or i == 0:
                print(f"正在处理第 {i + 1}/{len(label_files)} 个文件: {label_file}")

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
                    lines = f.readlines()
                    for line_num, line in enumerate(lines):
                        line = line.strip()
                        if line:
                            try:
                                # YOLO格式: class_id x_center y_center width height
                                parts = line.split()
                                if len(parts) >= 1:
                                    class_id = int(parts[0])
                                    if class_id in class_names:
                                        object_counts[class_id] += 1
                                    else:
                                        print(f"警告: 文件 {label_file} 第 {line_num + 1} 行包含未知类别ID: {class_id}")
                            except ValueError as e:
                                print(f"警告: 文件 {label_file} 第 {line_num + 1} 行格式错误: {line} - {e}")
                                continue

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
                error_count += 1
                continue

    print(f"\n处理完成!")
    print(f"成功生成metadata.jsonl文件: {metadata_file}")
    print(f"总共处理了 {processed_count} 组数据")
    print(f"错误文件数: {error_count}")

    # 验证生成的文件
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            actual_lines = sum(1 for _ in f)
        print(f"metadata.jsonl实际包含 {actual_lines} 行")

    return processed_count


# 运行脚本
if __name__ == "__main__":
    generate_metadata_jsonl()