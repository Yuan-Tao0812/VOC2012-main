import os
import json
import torch
from accelerate import Accelerator
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms

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
PRETRAINED_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # 修正的模型路径

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 优化训练参数以提高速度
BATCH_SIZE = 8  # A100可以支持更大batch size
EPOCHS = 12  # 减少epochs，先看效果
LR = 1e-4  # 保持官方推荐学习率
GRADIENT_ACCUMULATION_STEPS = 1
SCALE_LR = True
MAX_TOKEN_LENGTH = 77
IMAGE_SIZE = 512
CACHE_LATENTS = True  # 缓存latents以大幅提速
epoch_losses = []
weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 初始化Accelerator - 优化设置
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision="fp16" if torch.cuda.is_available() else "no",
    dataloader_config={
        "num_workers": 4,  # 增加数据加载并行度
        "pin_memory": True,
        "persistent_workers": True,  # 保持worker进程，减少重启开销
    }
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

# LoRA配置
unet_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

# 移动模型到设备
text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
vae = vae.to(accelerator.device, dtype=weight_dtype)
unet = unet.to(accelerator.device, dtype=weight_dtype)

# 添加LoRA适配器（官方方式）
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

# 优化器
optimizer = torch.optim.AdamW(
    lora_layers,
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-8,
)

# 学习率缩放（按官方逻辑）
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

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


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

        # 预缓存latents以提高训练速度
        if self.cache_latents:
            print("预缓存latents中...")
            self._cache_latents()

    def _cache_latents(self):
        """预缓存所有图片的latents"""
        self.cached_latents = []
        cache_dir = os.path.join(self.root_dir, "cached_latents")
        os.makedirs(cache_dir, exist_ok=True)

        for idx, item in enumerate(tqdm(self.entries, desc="缓存latents")):
            cache_file = os.path.join(cache_dir, f"latent_{idx}.pt")

            if os.path.exists(cache_file):
                # 加载已缓存的latent
                latent = torch.load(cache_file, map_location="cpu")
            else:
                # 生成并保存latent
                image_path = os.path.join(self.root_dir, item["image"])
                image = Image.open(image_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                # 编码为latent
                with torch.no_grad():
                    image_tensor = image.unsqueeze(0).to(self.vae.device, dtype=self.vae.dtype)
                    latent = self.vae.encode(image_tensor).latent_dist.sample()
                    latent = latent * self.vae.config.scaling_factor
                    latent = latent.squeeze(0).cpu()

                # 保存缓存
                torch.save(latent, cache_file)

            self.cached_latents.append(latent)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        if self.cache_latents:
            # 使用缓存的latents
            latent = self.cached_latents[idx]
            image = None  # 不需要原始图片
        else:
            # 原始方式
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


# 创建数据集和数据加载器 - 支持latent缓存
dataset = VisDroneControlNetDataset(
    DATA_DIR,
    PROMPT_FILE,
    tokenizer,
    vae=vae if CACHE_LATENTS else None,  # 传入VAE用于缓存
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
    num_workers=4,  # 增加worker数量
    pin_memory=True,
    persistent_workers=True,  # 保持worker进程
    prefetch_factor=2,  # 预取数据
)

# 学习率调度器
num_update_steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION_STEPS
max_train_steps = EPOCHS * num_update_steps_per_epoch

print(f"每epoch更新步数: {num_update_steps_per_epoch}")
print(f"总训练步数: {max_train_steps}")

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=max_train_steps,
)


# 官方保存方式的辅助函数
def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    return model


def save_lora_checkpoint(epoch, unet, optimizer, lr_scheduler, loss):
    """按照官方方式保存检查点（包含完整训练状态）"""
    # 1. 使用accelerator保存完整检查点（用于断点续训）
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-{epoch * len(dataloader)}")
    accelerator.save_state(checkpoint_path)

    # 2. 额外保存纯LoRA权重（用于推理）
    lora_path = os.path.join(CHECKPOINT_DIR, f"lora_epoch_{epoch}")
    os.makedirs(lora_path, exist_ok=True)

    unwrapped_unet = unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

    StableDiffusionPipeline.save_lora_weights(
        save_directory=lora_path,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=False,  # 匹配官方 .bin 格式
    )

    print(f"完整检查点保存到: {checkpoint_path}")
    print(f"LoRA权重保存到: {lora_path}")


def load_lora_checkpoint():
    """加载最新的检查点（Accelerator格式）"""
    start_epoch = 1

    # 查找accelerator格式的检查点
    checkpoints = []
    for item in os.listdir(CHECKPOINT_DIR):
        if item.startswith("checkpoint-"):
            step_num = int(item.split("-")[1])
            checkpoints.append((step_num, item))

    if checkpoints:
        # 按步数排序，取最新的
        checkpoints.sort(reverse=True)
        latest_checkpoint = checkpoints[0][1]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)

        print(f"发现最新检查点: {latest_checkpoint}")

        try:
            # 使用accelerator加载状态
            accelerator.load_state(checkpoint_path)

            # 计算对应的epoch
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


# 监控LoRA参数变化的函数
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


print(f"开始训练，从 epoch {start_epoch} 到 {EPOCHS}")
print(f"总训练步数: {max_train_steps}")
print(f"每epoch步数: {num_update_steps_per_epoch}")
print(f"使用批次大小: {BATCH_SIZE} (实际学习率已缩放)")
if CACHE_LATENTS:
    print("✅ 启用latent缓存，大幅提升训练速度")
print(f"预计每epoch时间: 约 45-60分钟 (6471张图)")
print(f"预计总训练时间: 约 {EPOCHS * 0.8:.1f}-{EPOCHS * 1.0:.1f} 小时")

# 训练循环
for epoch in range(start_epoch, EPOCHS + 1):
    unet.train()
    total_loss = 0.0

    # 训练前LoRA参数统计
    lora_stats_before = get_lora_param_stats(unet)
    if lora_stats_before:
        print(f"\nEpoch {epoch} 开始 - LoRA参数统计:")
        print(f"  均值: {lora_stats_before['mean']:.6f}")
        print(f"  标准差: {lora_stats_before['std']:.6f}")

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(unet):
            # 获取batch数据
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            # 编码文本
            with torch.no_grad():
                encoder_hidden_states = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )[0]

            # 获取latents - 使用缓存或实时编码
            if CACHE_LATENTS:
                latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
            else:
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

            # 添加噪声
            noise = torch.randn_like(latents, device=accelerator.device, dtype=weight_dtype)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=accelerator.device
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet前向传播
            model_pred = unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

            # 计算损失
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # 反向传播
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(lora_layers, 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 更新进度条
            current_loss = loss.detach().item()
            total_loss += current_loss
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                "step": f"{step}/{len(dataloader)}"
            })

    # epoch结束统计
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)

    # 训练后LoRA参数统计
    lora_stats_after = get_lora_param_stats(unet)
    if lora_stats_after and lora_stats_before:
        mean_change = abs(lora_stats_after['mean'] - lora_stats_before['mean'])
        print(f"\nEpoch {epoch} 结束:")
        print(f"  平均Loss: {avg_loss:.6f}")
        print(f"  LoRA参数均值变化: {mean_change:.8f}")

        if mean_change < 1e-8:
            print(f"  ⚠️  警告: LoRA参数变化极小，可能需要调整学习率或检查梯度")

    # 保存检查点
    if epoch % 5 == 0 or epoch == EPOCHS:
        save_lora_checkpoint(epoch, unet, optimizer, lr_scheduler, avg_loss)

print("训练完成!")

# 最终保存（官方方式）
print("保存最终LoRA模型...")
final_output_path = OUTPUT_DIR

# 获取LoRA权重（官方方式）
unwrapped_unet = unwrap_model(unet)
unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

# 保存LoRA权重（官方方式）- 保存为 .bin 格式以匹配官方示例
StableDiffusionPipeline.save_lora_weights(
    save_directory=final_output_path,
    unet_lora_layers=unet_lora_state_dict,
    safe_serialization=False,  # 保存为 pytorch_lora_weights.bin 格式
)

# 保存配置文件
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
        "final_loss": epoch_losses[-1] if epoch_losses else None
    }
}

with open(os.path.join(final_output_path, "training_config.json"), "w") as f:
    json.dump(config_info, f, indent=2)

# 绘制loss曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label="Average Training Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)

# Loss变化率
if len(epoch_losses) > 1:
    plt.subplot(1, 2, 2)
    loss_changes = [epoch_losses[i] - epoch_losses[i - 1] for i in range(1, len(epoch_losses))]
    plt.plot(range(2, len(epoch_losses) + 1), loss_changes, marker='s', color='orange', label="Loss Change",
             linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Change")
    plt.title("Loss Change Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
plt.show()

print(f"最终LoRA模型保存在: {final_output_path}")
print(f"主要文件:")
print(f"  - pytorch_lora_weights.bin (最终LoRA权重，用于推理)")
print(f"  - training_config.json (训练配置)")
print(f"  - training_curves.png (Loss曲线)")
print(f"\n检查点文件保存在: {CHECKPOINT_DIR}")
print(f"  - checkpoint-xxxx/ (完整训练状态，用于断点续训)")
print(f"  - lora_epoch_x/ (LoRA权重备份)")

print("\n" + "=" * 60)
print("🎨 开始推理测试...")
print("=" * 60)

# 清理GPU内存，为推理做准备
del unet, text_encoder, vae, optimizer, lr_scheduler
torch.cuda.empty_cache()

try:
    # 加载推理pipeline
    print("正在加载推理模型...")
    inference_pipeline = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,  # 关闭安全检查以节省内存
        requires_safety_checker=False
    ).to("cuda")

    # 加载训练好的LoRA权重
    print("正在加载LoRA权重...")
    inference_pipeline.load_lora_weights(final_output_path)

    # 测试prompts - 基于你的VisDrone数据集风格
    test_prompts = [
        "There are 5 pedestrians, 1 van, and 2 trucks in the image.",
        "There are 10 pedestrians, 3 cars, and 1 bus in the image.",
        "There are 2 pedestrians, 4 cars, and 2 motorcycles in the image.",
        "There are 15 pedestrians, 2 vans, 1 truck, and 3 cars in the image.",
        "There are 8 pedestrians, 1 van, and 5 cars in the image."
    ]

    # 生成测试图片
    inference_results_dir = os.path.join(OUTPUT_DIR, "inference_results")
    os.makedirs(inference_results_dir, exist_ok=True)

    print(f"正在生成 {len(test_prompts)} 张测试图片...")

    for i, prompt in enumerate(test_prompts):
        print(f"生成图片 {i + 1}/{len(test_prompts)}: {prompt}")

        # 生成图片
        with torch.no_grad():
            image = inference_pipeline(
                prompt,
                num_inference_steps=25,  # 适中的步数，平衡质量和速度
                guidance_scale=7.5,
                generator=torch.manual_seed(42 + i)  # 固定随机种子便于比较
            ).images[0]

        # 保存图片
        image_path = os.path.join(inference_results_dir, f"test_image_{i + 1}.png")
        image.save(image_path)

        # 保存prompt信息
        prompt_info = {
            "image": f"test_image_{i + 1}.png",
            "prompt": prompt,
            "inference_steps": 25,
            "guidance_scale": 7.5,
            "seed": 42 + i
        }

        info_path = os.path.join(inference_results_dir, f"test_image_{i + 1}_info.json")
        with open(info_path, "w") as f:
            json.dump(prompt_info, f, indent=2)

    # 创建结果总览图
    print("创建结果总览...")
    from PIL import Image as PILImage
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LoRA Training Results - Generated Images', fontsize=16, fontweight='bold')

    for i, prompt in enumerate(test_prompts):
        row = i // 3
        col = i % 3

        image_path = os.path.join(inference_results_dir, f"test_image_{i + 1}.png")
        img = PILImage.open(image_path)

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Test {i + 1}", fontsize=12, fontweight='bold')
        axes[row, col].text(0.02, 0.98, prompt, transform=axes[row, col].transAxes,
                            fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[row, col].axis('off')

    # 隐藏最后一个空的subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    overview_path = os.path.join(inference_results_dir, "results_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("🎉 推理测试完成！")
    print("=" * 60)
    print(f"📁 结果保存在: {inference_results_dir}")
    print(f"🖼️  生成的图片:")
    for i in range(len(test_prompts)):
        print(f"   - test_image_{i + 1}.png")
    print(f"📊 结果总览: results_overview.png")
    print(f"📋 每张图片都有对应的 _info.json 文件记录生成参数")

    print(f"\n🔍 训练效果分析:")
    print(f"请检查生成的图片是否:")
    print(f"  ✓ 包含prompt中指定的物体类型和数量")
    print(f"  ✓ 具有航拍/俯视角度的风格")
    print(f"  ✓ 画面清晰，物体可识别")
    print(f"  ✓ 整体风格与VisDrone数据集相似")

    # 清理推理pipeline内存
    del inference_pipeline
    torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ 推理测试失败: {e}")
    print("请检查:")
    print("  - GPU内存是否充足")
    print("  - LoRA权重文件是否正确保存")
    print("  - 基础模型是否能正常加载")

    # 保存错误信息供调试
    error_info = {
        "error": str(e),
        "error_type": type(e).__name__,
        "suggestion": "检查GPU内存和模型文件完整性"
    }

    error_path = os.path.join(OUTPUT_DIR, "inference_error.json")
    with open(error_path, "w") as f:
        json.dump(error_info, f, indent=2)

    print(f"错误信息已保存到: {error_path}")

print("\n🌙 训练和测试全部完成！可以安心睡觉了～")
print(f"起床后查看结果目录: {OUTPUT_DIR}")

print(f"\n现在的文件结构和官方 sayakpaul/sd-model-finetuned-lora-t4 完全一致！")
print(f"保存的是 pytorch_lora_weights.bin 格式，与官方示例完全相同。")