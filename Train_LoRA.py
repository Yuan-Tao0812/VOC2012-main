import os
import json
import torch
from accelerate import Accelerator
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
import random
import numpy as np
import time

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
)
from diffusers.training_utils import cast_training_params
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model_state_dict

# 配置路径
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/"
PROMPT_FILE = "prompt.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"
CHECKPOINT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints"
PRETRAINED_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 🚨 紧急修复参数
BATCH_SIZE = 2
EPOCHS = 15
LR = 1e-3  # 大幅提升学习率
GRADIENT_ACCUMULATION_STEPS = 1  # 禁用梯度累积
MAX_TOKEN_LENGTH = 77
IMAGE_SIZE = 512
MAX_GRAD_NORM = 10.0
SAVE_STEPS = 200

epoch_losses = []
step_losses = []
gradient_health_history = []
weight_dtype = torch.float32  # 强制float32


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# 初始化Accelerator - 禁用混合精度
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision="no",  # 禁用混合精度
)

print(f"🚨 紧急修复配置:")
print(f"设备: {accelerator.device}")
print(f"混合精度: {accelerator.mixed_precision}")
print(f"学习率: {LR}")

# 加载模型
print("加载预训练模型...")
noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="unet")

# 冻结模型
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# LoRA配置
unet_lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.0,
)

# 移动到设备
text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
vae = vae.to(accelerator.device, dtype=weight_dtype)
unet = unet.to(accelerator.device, dtype=weight_dtype)

# 添加LoRA
unet.add_adapter(unet_lora_config)

# 获取LoRA参数
lora_layers = []
for name, param in unet.named_parameters():
    if param.requires_grad:
        lora_layers.append(param)

if not lora_layers:
    raise RuntimeError("没有找到可训练的LoRA参数!")

print(f"找到 {len(lora_layers)} 个LoRA参数")

# 优化器
optimizer = torch.optim.AdamW(
    lora_layers,
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=1e-3,
    eps=1e-6,
)

# 数据变换
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class VisDroneDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, prompt_file, tokenizer, transform=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.transform = transform

        prompt_path = os.path.join(root_dir, prompt_file)
        with open(prompt_path, "r") as f:
            self.entries = [json.loads(line) for line in f]

        print(f"加载了 {len(self.entries)} 个样本")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        image_path = os.path.join(self.root_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tokenized = self.tokenizer(
            item["prompt"],
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "image": image,
        }


# 创建数据集
dataset = VisDroneDataset(DATA_DIR, PROMPT_FILE, tokenizer, transform)


def collate_fn(examples):
    return {
        "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
        "pixel_values": torch.stack([ex["image"] for ex in examples]),
    }


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)


def check_gradients(unet, step):
    """检查梯度状态"""
    total_grad_norm = 0
    valid_count = 0
    zero_count = 0

    for name, param in unet.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                if grad_norm == 0:
                    zero_count += 1
                else:
                    valid_count += 1
            else:
                zero_count += 1

    total_grad_norm = total_grad_norm ** 0.5

    print(f"Step {step}: 梯度范数={total_grad_norm:.6f}, 有效={valid_count}, 零={zero_count}")

    if total_grad_norm == 0:
        return "critical"
    elif total_grad_norm < 1e-6:
        return "too_small"
    else:
        return "healthy"


# 准备训练
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

print(f"开始训练...")

best_loss = float('inf')
global_step = 0

for epoch in range(1, EPOCHS + 1):
    unet.train()
    total_loss = 0.0
    epoch_steps = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

    for step, batch in enumerate(progress_bar):
        # 前向传播
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids, attention_mask)[0]
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # 添加噪声
        noise = torch.randn_like(latents, dtype=weight_dtype)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # UNet预测
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        # 计算损失
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        else:
            target = noise_scheduler.get_velocity(latents, noise, timesteps)

        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # 反向传播
        accelerator.backward(loss)

        # 梯度检查
        if global_step % 10 == 0:
            grad_status = check_gradients(unet, global_step)
            gradient_health_history.append((global_step, grad_status))

            # 自动调整学习率
            if grad_status == "critical":
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= 2.0
                    print(f"提高学习率: {old_lr:.2e} → {param_group['lr']:.2e}")

        # 优化
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(lora_layers, MAX_GRAD_NORM)

        optimizer.step()
        optimizer.zero_grad()

        # 记录
        current_loss = loss.detach().item()
        total_loss += current_loss
        step_losses.append(current_loss)
        epoch_steps += 1
        global_step += 1

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        # 保存检查点
        if global_step % SAVE_STEPS == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"emergency_{global_step}")
            os.makedirs(checkpoint_path, exist_ok=True)

            unwrapped_unet = accelerator.unwrap_model(unet)
            lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

            StableDiffusionPipeline.save_lora_weights(
                save_directory=checkpoint_path,
                unet_lora_layers=lora_state_dict,
                safe_serialization=False,
            )

    # Epoch结束
    avg_loss = total_loss / epoch_steps
    epoch_losses.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"\n新的最佳loss: {best_loss:.6f}")

    # 计算变化
    loss_change_pct = 0
    if len(epoch_losses) > 1:
        loss_change = epoch_losses[-2] - avg_loss
        loss_change_pct = (loss_change / epoch_losses[-2]) * 100

    print(f"\nEpoch {epoch} 完成:")
    print(f"  平均Loss: {avg_loss:.6f}")
    print(f"  变化: {loss_change_pct:+.2f}%")
    print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")

print("训练完成!")

# 保存最终模型
print("保存最终模型...")
unwrapped_unet = accelerator.unwrap_model(unet)
lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

StableDiffusionPipeline.save_lora_weights(
    save_directory=OUTPUT_DIR,
    unet_lora_layers=lora_state_dict,
    safe_serialization=False,
)

# 生成报告
report = {
    "training_completed": True,
    "epochs": len(epoch_losses),
    "final_loss": best_loss,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "gradient_checks": len(gradient_health_history),
    "healthy_gradients": sum(1 for _, status in gradient_health_history if status == "healthy")
}

with open(os.path.join(OUTPUT_DIR, "emergency_report.json"), "w") as f:
    json.dump(report, f, indent=2)

# 简单可视化
if len(step_losses) > 0:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(step_losses)
    plt.title("Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    if gradient_health_history:
        steps, statuses = zip(*gradient_health_history)
        status_values = [0 if s == "critical" else 1 if s == "too_small" else 2 for s in statuses]
        plt.scatter(steps, status_values)
        plt.yticks([0, 1, 2], ['Critical', 'Too Small', 'Healthy'])
        plt.title("Gradient Health")
        plt.xlabel("Step")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_summary.png"))
    plt.show()

print(f"✅ 紧急修复完成!")
print(f"模型保存在: {OUTPUT_DIR}")
print(f"最终loss: {best_loss:.6f}")

# 梯度健康总结
if gradient_health_history:
    healthy_count = sum(1 for _, status in gradient_health_history if status == "healthy")
    total_checks = len(gradient_health_history)
    print(f"梯度健康检查: {healthy_count}/{total_checks} ({healthy_count / total_checks * 100:.1f}%)")

    if healthy_count > 0:
        print("✅ 检测到健康梯度，修复成功!")
    else:
        print("❌ 仍未检测到健康梯度，需要进一步调试")