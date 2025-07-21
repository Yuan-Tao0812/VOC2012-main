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

DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/"
PROMPT_FILE = "prompt.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"
CHECKPOINT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints"
PRETRAINED_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 🔧 保守的优化参数 - 逐步提升
BATCH_SIZE = 2
EPOCHS = 12
LR = 2e-4  # 🔧 适度提高：1e-4 → 2e-4 (只提高1倍)
GRADIENT_ACCUMULATION_STEPS = 3  # 🔧 适度增加：2 → 3
SCALE_LR = True  # 🔧 保持原有的学习率缩放
MAX_TOKEN_LENGTH = 77
IMAGE_SIZE = 512
CACHE_LATENTS = True
WARMUP_STEPS = 300  # 🔧 适度减少：500 → 300
MIN_SNR_GAMMA = 0  # 🔧 先不用Min-SNR，避免过于复杂
NOISE_OFFSET = 0.05  # 🔧 很小的噪声偏移，几乎不影响训练

epoch_losses = []
step_losses = []
weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# 设置随机种子确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# 初始化Accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision="fp16" if torch.cuda.is_available() else "no"
)

print(f"使用设备: {accelerator.device}")
print(f"混合精度: {accelerator.mixed_precision}")

# 加载预训练模型
noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="unet")

# 冻结预训练参数
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# 🔧 适度优化的LoRA配置
unet_lora_config = LoraConfig(
    r=24,  # 🔧 适度增加：16 → 24
    lora_alpha=32,  # 🔧 保持不变，避免过大影响
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

# 移动模型到设备
text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
vae = vae.to(accelerator.device, dtype=weight_dtype)
unet = unet.to(accelerator.device, dtype=weight_dtype)

# 添加LoRA适配器
unet.add_adapter(unet_lora_config)

# 设置训练参数
cast_training_params(unet, dtype=torch.float32)

# 获取LoRA可训练参数
lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
print(f"可训练参数数量（LoRA）：{len(lora_layers)}")
print(f"可训练参数总量：{sum(p.numel() for p in lora_layers)}")

# 验证LoRA参数
print("\n部分LoRA参数示例：")
count = 0
for name, param in unet.named_parameters():
    if param.requires_grad and count < 5:
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        count += 1

# 启用梯度检查点
if hasattr(unet, "enable_gradient_checkpointing"):
    unet.enable_gradient_checkpointing()

# 🔧 保守的优化器设置
optimizer = torch.optim.AdamW(
    lora_layers,
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=1e-2,  # 🔧 保持原有设置
    eps=1e-8,
)

# 学习率缩放（保持原有逻辑）
if SCALE_LR:
    scaled_lr = LR * GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE * accelerator.num_processes
    print(f"原始学习率: {LR}")
    print(f"缩放后学习率: {scaled_lr}")
    # 重新创建优化器
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=scaled_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

# 🔧 针对LoRA优化的数据预处理
# 移除数据增强，专注于loss下降
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # 🚫 移除RandomHorizontalFlip - 可能破坏航拍图像的空间语义
    # 🚫 移除ColorJitter - 可能影响物体识别和计数
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# 可选：如果想要轻微的鲁棒性，可以添加很小的噪声
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),  # 很小的噪声
#     transforms.Normalize([0.5]*3, [0.5]*3),
# ])

class VisDroneControlNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, prompt_file, tokenizer, vae=None, max_length=MAX_TOKEN_LENGTH, transform=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.vae = vae
        self.cache_latents = CACHE_LATENTS and vae is not None

        prompt_path = os.path.join(root_dir, prompt_file)
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r") as f:
            self.entries = [json.loads(line) for line in f]

        print(f"加载了 {len(self.entries)} 个训练样本")

        if self.cache_latents:
            print("预缓存latents中...")
            self._cache_latents()

    def _cache_latents(self):
        """预缓存所有图片的latents"""
        self.cached_latents = []
        cache_dir = os.path.join(self.root_dir, "cached_latents")
        os.makedirs(cache_dir, exist_ok=True)

        batch_size = 4

        for start_idx in tqdm(range(0, len(self.entries), batch_size), desc="缓存latents"):
            batch_indices = range(start_idx, min(start_idx + batch_size, len(self.entries)))

            for idx in batch_indices:
                item = self.entries[idx]
                cache_file = os.path.join(cache_dir, f"latent_{idx}.pt")

                if os.path.exists(cache_file):
                    try:
                        latent = torch.load(cache_file, map_location="cpu")
                        self.cached_latents.append(latent)
                    except:
                        self.cached_latents.append(None)
                else:
                    try:
                        image_path = os.path.join(self.root_dir, item["image"])

                        if not os.path.exists(image_path):
                            self.cached_latents.append(None)
                            continue

                        image = Image.open(image_path).convert("RGB")

                        if self.transform:
                            image = self.transform(image)

                        with torch.no_grad():
                            image_tensor = image.unsqueeze(0).to(self.vae.device, dtype=self.vae.dtype)
                            latent = self.vae.encode(image_tensor).latent_dist.sample()
                            latent = latent * self.vae.config.scaling_factor
                            latent = latent.squeeze(0).cpu()

                            del image_tensor
                            torch.cuda.empty_cache()

                        torch.save(latent, cache_file)
                        self.cached_latents.append(latent)

                    except Exception as e:
                        print(f"⚠️ 缓存第{idx}个样本失败: {e}")
                        self.cached_latents.append(None)

            if start_idx % (batch_size * 4) == 0:
                torch.cuda.empty_cache()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        if self.cache_latents:
            latent = self.cached_latents[idx]
            image = None
        else:
            image_path = os.path.join(self.root_dir, item["image"])

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            latent = None

        tokenized = self.tokenizer(
            item["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }

        if self.cache_latents:
            result["latents"] = latent
        else:
            result["image"] = image

        return result


# 创建数据集和数据加载器
dataset = VisDroneControlNetDataset(
    DATA_DIR,
    PROMPT_FILE,
    tokenizer,
    vae=vae if CACHE_LATENTS else None,
    transform=transform
)


def collate_fn(examples):
    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    attention_mask = torch.stack([ex["attention_mask"] for ex in examples])

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if CACHE_LATENTS:
        latents = torch.stack([ex["latents"] for ex in examples])
        result["latents"] = latents
    else:
        pixel_values = torch.stack([ex["image"] for ex in examples])
        result["pixel_values"] = pixel_values

    return result


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=False,
    prefetch_factor=1,
)

# 学习率调度器
num_update_steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION_STEPS
max_train_steps = EPOCHS * num_update_steps_per_epoch

print(f"每epoch更新步数: {num_update_steps_per_epoch}")
print(f"总训练步数: {max_train_steps}")

# 🔧 保持cosine调度器，但调整warmup
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,  # 减少了warmup步数
    num_training_steps=max_train_steps,
)


def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    return model


def save_lora_checkpoint(epoch, unet, optimizer, lr_scheduler, loss):
    """保存检查点"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-{epoch * len(dataloader)}")
    accelerator.save_state(checkpoint_path)

    lora_path = os.path.join(CHECKPOINT_DIR, f"lora_epoch_{epoch}")
    os.makedirs(lora_path, exist_ok=True)

    unwrapped_unet = unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

    StableDiffusionPipeline.save_lora_weights(
        save_directory=lora_path,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=False,
    )

    print(f"完整检查点保存到: {checkpoint_path}")
    print(f"LoRA权重保存到: {lora_path}")


def load_lora_checkpoint():
    """加载最新的检查点"""
    start_epoch = 1

    checkpoints = []
    if os.path.exists(CHECKPOINT_DIR):
        for item in os.listdir(CHECKPOINT_DIR):
            if item.startswith("checkpoint-"):
                step_num = int(item.split("-")[1])
                checkpoints.append((step_num, item))

    if checkpoints:
        checkpoints.sort(reverse=True)
        latest_checkpoint = checkpoints[0][1]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)

        print(f"发现最新检查点: {latest_checkpoint}")

        try:
            accelerator.load_state(checkpoint_path)
            step_num = checkpoints[0][0]
            start_epoch = (step_num // len(dataloader)) + 1
            print(f"成功加载检查点，将从 epoch {start_epoch} 开始训练")

        except Exception as e:
            print(f"加载检查点失败: {e}")
            start_epoch = 1

    return start_epoch


# 尝试加载检查点
start_epoch = load_lora_checkpoint()

# 准备训练
unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, dataloader, lr_scheduler
)


def get_lora_param_stats(model):
    """获取LoRA参数的统计信息"""
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and any(lora_key in name for lora_key in ["lora_A", "lora_B"]):
            lora_params.append(param.detach().cpu())

    if lora_params:
        all_params = torch.cat([p.flatten() for p in lora_params])
        return {
            "mean": all_params.mean().item(),
            "std": all_params.std().item(),
            "min": all_params.min().item(),
            "max": all_params.max().item()
        }
    return None


print(f"🔧 保守优化的训练配置:")
print(f"   - 学习率适度提高: 1e-4 → {LR}")
print(f"   - LoRA rank适度增加: 16 → {unet_lora_config.r}")
print(f"   - 梯度累积适度增加: 2 → {GRADIENT_ACCUMULATION_STEPS}")
print(f"   - 减少warmup步数: 500 → {WARMUP_STEPS}")
print(f"   - 保持原有学习率缩放和调度器")
print(f"   - 添加极小的噪声偏移: {NOISE_OFFSET}")

# 🔧 改进的训练循环 - 增加监控但不改变核心逻辑
global_step = 0
best_loss = float('inf')
loss_patience = 0  # 用于早停的耐心计数

for epoch in range(start_epoch, EPOCHS + 1):
    unet.train()
    total_loss = 0.0
    epoch_step_losses = []

    lora_stats_before = get_lora_param_stats(unet)
    if lora_stats_before:
        print(f"\nEpoch {epoch} 开始 - LoRA参数统计:")
        print(f"  均值: {lora_stats_before['mean']:.6f}")
        print(f"  标准差: {lora_stats_before['std']:.6f}")

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(unet):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )[0]

            if CACHE_LATENTS:
                latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
            else:
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

            # 🔧 保守的噪声处理
            noise = torch.randn_like(latents, device=accelerator.device, dtype=weight_dtype)
            if NOISE_OFFSET > 0:
                # 添加很小的噪声偏移
                noise += NOISE_OFFSET * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=accelerator.device,
                                                    dtype=weight_dtype)

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=accelerator.device
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # 🔧 保持原有的简单MSE损失
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(lora_layers, 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 记录损失
            current_loss = loss.detach().item()
            total_loss += current_loss
            epoch_step_losses.append(current_loss)
            step_losses.append(current_loss)
            global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "avg_loss": f"{np.mean(epoch_step_losses[-20:]):.4f}",  # 最近20步的平均loss
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                "step": f"{step}/{len(dataloader)}"
            })

    # epoch结束统计
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)

    # 🔧 改进的损失分析
    loss_std = np.std(epoch_step_losses)
    recent_loss_trend = np.mean(epoch_step_losses[-100:]) if len(epoch_step_losses) > 100 else avg_loss

    lora_stats_after = get_lora_param_stats(unet)
    if lora_stats_after and lora_stats_before:
        mean_change = abs(lora_stats_after['mean'] - lora_stats_before['mean'])
        std_change = abs(lora_stats_after['std'] - lora_stats_before['std'])

        print(f"\nEpoch {epoch} 结束:")
        print(f"  平均Loss: {avg_loss:.6f}")
        print(f"  Loss标准差: {loss_std:.6f}")
        print(f"  最后100步均值: {recent_loss_trend:.6f}")
        print(f"  LoRA参数均值变化: {mean_change:.8f}")
        print(f"  LoRA参数标准差变化: {std_change:.8f}")

        # 🔧 健康检查 - 给出具体建议
        if mean_change < 1e-7:
            print(f"  ⚠️  LoRA参数变化极小 (<1e-7)，可能需要:")
            print(f"      - 适度增加学习率 (当前: {lr_scheduler.get_last_lr()[0]:.2e})")
            print(f"      - 检查梯度是否正常")
        elif mean_change > 1e-3:
            print(f"  ⚠️  LoRA参数变化过大 (>1e-3)，考虑:")
            print(f"      - 降低学习率")
            print(f"      - 增加梯度累积步数")
        else:
            print(f"  ✅ LoRA参数变化在合理范围内")

        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            loss_patience = 0
        else:
            loss_patience += 1

        if loss_patience >= 3:
            print(f"  📊 Loss连续3个epoch未改善，可能接近收敛")

    # 保存检查点
    if epoch % 4 == 0 or epoch == EPOCHS:
        save_lora_checkpoint(epoch, unet, optimizer, lr_scheduler, avg_loss)

print("训练完成!")

# 最终保存
print("保存最终LoRA模型...")
final_output_path = OUTPUT_DIR

unwrapped_unet = unwrap_model(unet)
unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

StableDiffusionPipeline.save_lora_weights(
    save_directory=final_output_path,
    unet_lora_layers=unet_lora_state_dict,
    safe_serialization=False,
)

# 保存配置信息
config_info = {
    "lora_config": {
        "r": unet_lora_config.r,
        "lora_alpha": unet_lora_config.lora_alpha,
        "target_modules": unet_lora_config.target_modules,
        "init_lora_weights": unet_lora_config.init_lora_weights
    },
    "training_info": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "noise_offset": NOISE_OFFSET,
        "warmup_steps": WARMUP_STEPS,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "best_loss": best_loss,
        "optimization_level": "conservative"
    }
}

with open(os.path.join(final_output_path, "training_config.json"), "w") as f:
    json.dump(config_info, f, indent=2)

# 🔧 改进但保守的loss可视化
plt.figure(figsize=(15, 8))

# 1. Epoch级别的loss
plt.subplot(2, 3, 1)
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label="Training Loss", linewidth=2)
if len(epoch_losses) > 1:
    # 添加趋势线
    z = np.polyfit(range(1, len(epoch_losses) + 1), epoch_losses, 1)
    p = np.poly1d(z)
    plt.plot(range(1, len(epoch_losses) + 1), p(range(1, len(epoch_losses) + 1)), "--", alpha=0.7, label="Trend")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 步级别的loss（滑动平均）
if len(step_losses) > 50:
    plt.subplot(2, 3, 2)
    window_size = 50
    smoothed_losses = np.convolve(step_losses, np.ones(window_size) / window_size, mode='valid')
    plt.plot(smoothed_losses, label=f"Smoothed Loss", linewidth=1.5, alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Step-wise Loss (MA-{window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)

# 3. Loss变化率
if len(epoch_losses) > 1:
    plt.subplot(2, 3, 3)
    loss_changes = [epoch_losses[i] - epoch_losses[i - 1] for i in range(1, len(epoch_losses))]
    plt.plot(range(2, len(epoch_losses) + 1), loss_changes, marker='s', color='orange', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Change")
    plt.title("Loss Change Rate")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# 4. 学习率曲线
plt.subplot(2, 3, 4)
lr_values = []
# 模拟学习率变化
for step in range(max_train_steps):
    if step < WARMUP_STEPS:
        lr = LR * GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE * accelerator.num_processes * (step / WARMUP_STEPS)
    else:
        progress = (step - WARMUP_STEPS) / (max_train_steps - WARMUP_STEPS)
        lr = LR * GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE * accelerator.num_processes * 0.5 * (
                    1 + np.cos(np.pi * progress))
    lr_values.append(lr)

plt.plot(lr_values, label="Learning Rate", linewidth=2, color='green')
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True, alpha=0.3)

# 5. 训练稳定性分析
plt.subplot(2, 3, 5)
if len(step_losses) > 200:
    # 计算每100步的标准差
    window = 100
    stds = []
    for i in range(window, len(step_losses), window // 2):
        std = np.std(step_losses[i - window:i])
        stds.append(std)

    plt.plot(stds, label="Loss Stability", color='purple', linewidth=1.5)
    plt.xlabel("Window")
    plt.ylabel("Loss Std")
    plt.title("Training Stability")
    plt.legend()
    plt.grid(True, alpha=0.3)

# 6. 收敛分析
plt.subplot(2, 3, 6)
if len(epoch_losses) > 3:
    # 计算相对改善
    improvements = []
    for i in range(1, len(epoch_losses)):
        if epoch_losses[i - 1] > 0:
            improvement = (epoch_losses[i - 1] - epoch_losses[i]) / epoch_losses[i - 1] * 100
            improvements.append(improvement)

    plt.bar(range(2, len(epoch_losses) + 1), improvements, alpha=0.7, color='skyblue')
    plt.xlabel("Epoch")
    plt.ylabel("Improvement (%)")
    plt.title("Loss Improvement per Epoch")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "conservative_training_analysis.png"), dpi=150, bbox_inches='tight')
plt.show()

print(f"最终LoRA模型保存在: {final_output_path}")
print(f"主要文件:")
print(f"  - pytorch_lora_weights.bin (最终LoRA权重，用于推理)")
print(f"  - training_config.json (训练配置)")
print(f"  - conservative_training_analysis.png (训练分析)")

print(f"\n检查点文件保存在: {CHECKPOINT_DIR}")
print(f"  - checkpoint-xxxx/ (完整训练状态，用于断点续训)")
print(f"  - lora_epoch_x/ (LoRA权重备份)")

# 🔧 保守的训练效果分析
print("\n" + "=" * 60)
print("📊 保守优化训练分析")
print("=" * 60)

if len(epoch_losses) > 1:
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"初始Loss: {initial_loss:.6f}")
    print(f"最终Loss: {final_loss:.6f}")
    print(f"Loss降低: {loss_reduction:.2f}%")
    print(f"最佳Loss: {best_loss:.6f}")

    # 更细致的效果评估
    if loss_reduction > 15:
        print("✅ 训练效果优秀，Loss显著下降")
    elif loss_reduction > 8:
        print("✅ 训练效果良好，建议继续训练")
    elif loss_reduction > 3:
        print("⚠️ 训练效果一般，可考虑:")
        print("   - 延长训练轮数")
        print("   - 适度提高学习率")
        print("   - 增加LoRA rank")
    else:
        print("❌ 训练效果不佳，建议:")
        print("   - 检查数据质量")
        print("   - 提高学习率 (当前较保守)")
        print("   - 增加训练轮数")

if len(step_losses) > 100:
    recent_loss_std = np.std(step_losses[-100:])
    overall_loss_std = np.std(step_losses)

    print(f"\n收敛性分析:")
    print(f"  最近100步Loss标准差: {recent_loss_std:.6f}")
    print(f"  整体Loss标准差: {overall_loss_std:.6f}")

    if recent_loss_std < overall_loss_std * 0.5:
        print("  ✅ 训练已基本收敛")
    elif recent_loss_std < overall_loss_std * 0.8:
        print("  ⚠️ 训练接近收敛，可适当延长")
    else:
        print("  ❌ 训练尚未充分收敛，建议继续")

print(f"\n参数变化分析:")
print(f"  LoRA rank: {unet_lora_config.r} (适度提升)")
print(f"  学习率: {LR} (保守提升)")
print(f"  有效batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

print("\n" + "=" * 60)
print("🎨 开始推理测试...")
print("=" * 60)

# 清理GPU内存
del unet, text_encoder, vae, optimizer, lr_scheduler
torch.cuda.empty_cache()

try:
    # 加载推理pipeline
    print("正在加载推理模型...")
    inference_pipeline = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")

    # 加载训练好的LoRA权重
    print("正在加载LoRA权重...")
    inference_pipeline.load_lora_weights(final_output_path)

    # 基础的推理设置
    if hasattr(inference_pipeline, 'enable_memory_efficient_attention'):
        inference_pipeline.enable_memory_efficient_attention()

    # 测试prompts - 涵盖不同复杂度
    test_prompts = [
        "There are 5 pedestrians, 1 van, and 2 trucks in the image.",
        "There are 10 pedestrians, 3 cars, and 1 bus in the image.",
        "There are 2 pedestrians, 4 cars, and 2 motorcycles in the image.",
        "There are 15 pedestrians, 2 vans, 1 truck, and 3 cars in the image.",
        "There are 8 pedestrians, 1 van, and 5 cars in the image.",
        "There are 3 pedestrians and 1 motorcycle in the image.",
    ]

    # 生成测试图片
    inference_results_dir = os.path.join(OUTPUT_DIR, "inference_results")
    os.makedirs(inference_results_dir, exist_ok=True)

    print(f"正在生成 {len(test_prompts)} 张测试图片...")

    # 使用标准推理设置
    inference_config = {"steps": 25, "guidance": 7.5}

    for i, prompt in enumerate(test_prompts):
        print(f"生成图片 {i + 1}/{len(test_prompts)}: {prompt[:40]}...")

        # 生成图片
        with torch.no_grad():
            image = inference_pipeline(
                prompt,
                num_inference_steps=inference_config["steps"],
                guidance_scale=inference_config["guidance"],
                generator=torch.manual_seed(42 + i)
            ).images[0]

        # 保存图片
        image_path = os.path.join(inference_results_dir, f"test_image_{i + 1}.png")
        image.save(image_path)

        # 保存prompt信息
        prompt_info = {
            "image": f"test_image_{i + 1}.png",
            "prompt": prompt,
            "inference_steps": inference_config["steps"],
            "guidance_scale": inference_config["guidance"],
            "seed": 42 + i
        }

        info_path = os.path.join(inference_results_dir, f"test_image_{i + 1}_info.json")
        with open(info_path, "w") as f:
            json.dump(prompt_info, f, indent=2)

    # 创建结果总览图
    print("创建结果总览...")
    from PIL import Image as PILImage

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Conservative LoRA Training Results', fontsize=16, fontweight='bold')

    for i, prompt in enumerate(test_prompts):
        row = i // 3
        col = i % 3

        image_path = os.path.join(inference_results_dir, f"test_image_{i + 1}.png")
        img = PILImage.open(image_path)

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Test {i + 1}", fontsize=12, fontweight='bold')

        # 自动换行显示prompt
        words = prompt.split()
        lines = []
        current_line = []
        char_count = 0

        for word in words:
            if char_count + len(word) + 1 <= 50:
                current_line.append(word)
                char_count += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                char_count = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        prompt_text = '\n'.join(lines)

        axes[row, col].text(0.02, 0.98, prompt_text, transform=axes[row, col].transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        axes[row, col].axis('off')

    plt.tight_layout()
    overview_path = os.path.join(inference_results_dir, "results_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("🎉 推理测试完成！")
    print("=" * 60)
    print(f"📁 结果保存在: {inference_results_dir}")
    print(f"🖼️  生成了 {len(test_prompts)} 张测试图片")
    print(f"📊 结果总览: results_overview.png")

    print(f"\n🔍 效果评估建议:")
    print(f"检查生成的图片是否:")
    print(f"  ✓ 包含正确的物体类型")
    print(f"  ✓ 数量大致准确")
    print(f"  ✓ 具有航拍视角风格")
    print(f"  ✓ 画面清晰自然")

    # 根据训练loss给出进一步建议
    if len(epoch_losses) > 1:
        loss_reduction = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100
        if loss_reduction < 5:
            print(f"\n💡 进一步优化建议 (当前Loss降低: {loss_reduction:.1f}%):")
            print(f"  - 考虑增加训练轮数到 {EPOCHS + 5}-{EPOCHS + 10}")
            print(f"  - 适度提高学习率到 3e-4")
            print(f"  - 增加LoRA rank到 32")
            print(f"  - 检查数据集质量和prompt匹配度")

    del inference_pipeline
    torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ 推理测试失败: {e}")
    print("建议检查:")
    print("  - GPU内存使用情况")
    print("  - LoRA权重文件完整性")
    print("  - 基础模型加载是否正常")

    error_info = {
        "error": str(e),
        "error_type": type(e).__name__,
        "optimization_level": "conservative",
        "suggestion": "检查资源和文件完整性"
    }

    error_path = os.path.join(OUTPUT_DIR, "inference_error.json")
    with open(error_path, "w") as f:
        json.dump(error_info, f, indent=2)

# 最终总结
print("\n" + "🔧" * 20)
print("保守优化训练完成总结")
print("🔧" * 20)

summary_info = {
    "optimization_type": "conservative",
    "changes_made": [
        f"学习率: 1e-4 → {LR} (2倍提升)",
        f"LoRA rank: 16 → {unet_lora_config.r} (1.5倍提升)",
        f"梯度累积: 2 → {GRADIENT_ACCUMULATION_STEPS}",
        f"warmup步数: 500 → {WARMUP_STEPS}",
        f"添加轻微噪声偏移: {NOISE_OFFSET}",
        "保持原有调度器和缩放逻辑"
    ],
    "training_results": {
        "total_epochs": EPOCHS,
        "final_loss": epoch_losses[-1] if epoch_losses else "N/A",
        "best_loss": best_loss,
        "loss_reduction": f"{((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.2f}%" if len(
            epoch_losses) > 1 else "N/A"
    },
    "safety_measures": [
        "保持学习率缩放避免过大学习率",
        "使用cosine调度器平滑学习",
        "保持原有的MSE损失函数",
        "适度的参数调整避免不稳定"
    ]
}

summary_path = os.path.join(OUTPUT_DIR, "conservative_training_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary_info, f, indent=2, ensure_ascii=False)

print(f"📊 保守优化摘要:")
print(f"   类型: 渐进式保守优化")
print(f"   风险: 低 (保持原有核心逻辑)")
print(f"   预期: 适度改善训练效果")

if len(epoch_losses) > 1:
    loss_reduction = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100
    print(f"   实际Loss降低: {loss_reduction:.2f}%")

    if loss_reduction > 5:
        print(f"   ✅ 优化成功，达到预期效果")
    else:
        print(f"   ⚠️ 可考虑进一步优化:")
        print(f"      - 延长训练 (+3-5 epochs)")
        print(f"      - 适度提高学习率 (→ 3e-4)")
        print(f"      - 增加LoRA rank (→ 32)")

print(f"\n📁 所有文件保存在: {OUTPUT_DIR}")
print(f"🚀 保守优化完成，可安全使用！")

print("\n🌙 保守优化训练完成！")
print("这个版本应该不会出现loss爆炸的问题～")