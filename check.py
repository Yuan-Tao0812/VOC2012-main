"""
LoRA训练质量诊断脚本
用于检查缓存质量、数据完整性和训练状态
"""

import os
import torch
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置路径
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/"
CACHE_DIR = os.path.join(DATA_DIR, "cached_latents")
CHECKPOINT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"


def check_cache_quality():
    """检查latent缓存的质量"""
    print("🔍 检查Latent缓存质量...")

    if not os.path.exists(CACHE_DIR):
        print("❌ 缓存目录不存在")
        return False

    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pt')]
    print(f"📊 缓存文件数量: {len(cache_files)}")

    # 检查文件大小分布
    file_sizes = []
    corrupted_files = []
    valid_samples = []

    print("📏 分析缓存文件大小...")
    for i, file in enumerate(tqdm(cache_files[:100], desc="检查缓存")):  # 检查前100个
        file_path = os.path.join(CACHE_DIR, file)
        size = os.path.getsize(file_path)
        file_sizes.append(size)

        # 尝试加载文件
        try:
            latent = torch.load(file_path, map_location="cpu")
            if latent is not None and latent.numel() > 0:
                valid_samples.append({
                    'file': file,
                    'size': size,
                    'shape': latent.shape,
                    'mean': latent.mean().item(),
                    'std': latent.std().item(),
                    'min': latent.min().item(),
                    'max': latent.max().item()
                })
            else:
                corrupted_files.append(file)
        except Exception as e:
            corrupted_files.append(file)
            print(f"⚠️ 损坏的缓存文件: {file}, 错误: {e}")

    # 统计分析
    file_sizes = np.array(file_sizes)
    print(f"\n📈 缓存文件大小统计:")
    print(f"  平均大小: {file_sizes.mean():.0f} bytes")
    print(f"  最小大小: {file_sizes.min()} bytes")
    print(f"  最大大小: {file_sizes.max()} bytes")
    print(f"  标准差: {file_sizes.std():.0f} bytes")

    # 检查异常文件
    small_files = (file_sizes < 1000).sum()  # 小于1KB的文件
    print(f"⚠️ 异常小文件(<1KB): {small_files}")
    print(f"❌ 损坏文件: {len(corrupted_files)}")
    print(f"✅ 有效文件: {len(valid_samples)}")

    # 分析有效样本的latent统计
    if valid_samples:
        print(f"\n📊 Latent数据统计 (基于{len(valid_samples)}个样本):")
        means = [s['mean'] for s in valid_samples]
        stds = [s['std'] for s in valid_samples]

        print(f"  Latent均值范围: {min(means):.4f} ~ {max(means):.4f}")
        print(f"  Latent标准差范围: {min(stds):.4f} ~ {max(stds):.4f}")

        # 检查是否有异常值
        mean_std = np.std(means)
        std_std = np.std(stds)
        print(f"  均值的标准差: {mean_std:.4f} (应该<0.5)")
        print(f"  标准差的标准差: {std_std:.4f} (应该<0.5)")

        # 判断质量
        if small_files > 10 or len(corrupted_files) > 5:
            print("❌ 缓存质量差：建议重新缓存")
            return False
        elif mean_std > 0.5 or std_std > 0.5:
            print("⚠️ 缓存数据异常：可能有数据不一致")
            return False
        else:
            print("✅ 缓存质量良好")
            return True
    else:
        print("❌ 没有有效的缓存文件")
        return False


def check_data_integrity():
    """检查原始数据完整性"""
    print("\n🔍 检查原始数据完整性...")

    # 检查prompt文件
    prompt_file = os.path.join(DATA_DIR, "prompt.jsonl")
    if not os.path.exists(prompt_file):
        print("❌ prompt.jsonl文件不存在")
        return False

    # 读取prompt数据
    try:
        with open(prompt_file, "r") as f:
            entries = [json.loads(line) for line in f]
        print(f"📄 Prompt条目数量: {len(entries)}")
    except Exception as e:
        print(f"❌ 读取prompt文件失败: {e}")
        return False

    # 检查图片文件存在性
    missing_images = []
    existing_images = []

    print("🖼️ 检查图片文件存在性...")
    for i, entry in enumerate(tqdm(entries[:200], desc="检查图片")):  # 检查前200个
        image_path = os.path.join(DATA_DIR, entry["image"])
        if os.path.exists(image_path):
            try:
                # 尝试打开图片
                img = Image.open(image_path)
                existing_images.append(entry["image"])
                img.close()
            except Exception as e:
                missing_images.append(f"{entry['image']} (损坏: {e})")
        else:
            missing_images.append(entry["image"])

    print(f"✅ 有效图片: {len(existing_images)}")
    print(f"❌ 缺失/损坏图片: {len(missing_images)}")

    if len(missing_images) > 10:
        print("⚠️ 缺失图片过多，可能影响训练质量")
        print("前10个缺失图片:")
        for img in missing_images[:10]:
            print(f"  - {img}")
        return False
    else:
        print("✅ 数据完整性良好")
        return True


def check_training_progress():
    """检查训练进度和模型状态"""
    print("\n🔍 检查训练进度...")

    # 检查是否有保存的模型
    checkpoints = []
    if os.path.exists(CHECKPOINT_DIR):
        for item in os.listdir(CHECKPOINT_DIR):
            if item.startswith("balanced_checkpoint_epoch_"):
                epoch_num = item.split("_")[-1]
                checkpoints.append(int(epoch_num))

    if checkpoints:
        latest_epoch = max(checkpoints)
        print(f"📁 找到检查点: epoch {sorted(checkpoints)}")
        print(f"🎯 最新检查点: epoch {latest_epoch}")

        # 检查最新检查点
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, f"balanced_checkpoint_epoch_{latest_epoch}")
        lora_weights_path = os.path.join(latest_checkpoint, "pytorch_lora_weights.bin")

        if os.path.exists(lora_weights_path):
            size = os.path.getsize(lora_weights_path)
            print(f"✅ LoRA权重文件大小: {size / 1024 / 1024:.2f} MB")

            # 尝试加载权重检查
            try:
                weights = torch.load(lora_weights_path, map_location="cpu")
                print(f"✅ LoRA权重层数: {len(weights)}")

                # 检查权重统计
                all_weights = torch.cat([w.flatten() for w in weights.values()])
                print(f"📊 权重统计:")
                print(f"  均值: {all_weights.mean().item():.6f}")
                print(f"  标准差: {all_weights.std().item():.6f}")
                print(f"  绝对值均值: {all_weights.abs().mean().item():.6f}")

                if all_weights.abs().mean().item() < 1e-6:
                    print("⚠️ 权重过小，可能训练未生效")
                    return False
                else:
                    print("✅ 权重正常")
                    return True

            except Exception as e:
                print(f"❌ 加载权重失败: {e}")
                return False
        else:
            print("❌ 找不到LoRA权重文件")
            return False
    else:
        print("📝 未找到检查点（第一轮训练）")
        return True


def analyze_loss_pattern(first_epoch_loss):
    """分析loss模式"""
    print(f"\n📈 分析第一轮平均Loss: {first_epoch_loss:.6f}")

    # 判断loss是否正常
    if first_epoch_loss < 0.01:
        print("❌ Loss过低，可能训练数据有问题")
        return False
    elif first_epoch_loss > 0.5:
        print("❌ Loss过高，可能学习率有问题或数据异常")
        return False
    elif 0.08 <= first_epoch_loss <= 0.25:
        print("✅ Loss在正常范围内")
        return True
    else:
        print("⚠️ Loss略异常，但可能可以接受")
        return True


def generate_diagnostic_report(first_epoch_loss=None):
    """生成完整的诊断报告"""
    print("🏥 开始全面诊断...")
    print("=" * 60)

    results = {
        "cache_quality": check_cache_quality(),
        "data_integrity": check_data_integrity(),
        "training_progress": check_training_progress()
    }

    if first_epoch_loss is not None:
        results["loss_analysis"] = analyze_loss_pattern(first_epoch_loss)

    print("\n" + "=" * 60)
    print("📋 诊断报告总结:")
    print("=" * 60)

    all_good = True
    for check, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check.replace('_', ' ').title()}: {status}")
        if not result:
            all_good = False

    print("=" * 60)
    if all_good:
        print("🎉 整体评估: 训练状态良好，可以继续！")
        print("✅ 建议: 继续当前训练，无需重新缓存")
        return "continue"
    else:
        print("⚠️ 整体评估: 发现问题，建议修复")
        print("🔧 建议: 考虑禁用缓存重新训练")
        return "restart"


# 使用方法
if __name__ == "__main__":
    print("🔍 LoRA训练诊断工具")
    print("请在第一轮训练完成后运行此脚本")
    print("\n如果要包含loss分析，请这样调用:")
    print("result = generate_diagnostic_report(first_epoch_loss=你的平均loss)")
    print("\n如果只想检查缓存和数据:")
    print("result = generate_diagnostic_report()")