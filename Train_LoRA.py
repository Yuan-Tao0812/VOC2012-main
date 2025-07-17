import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig
from diffusers import (
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.training_utils import cast_training_params

# === 配置参数 ===
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/"
PROMPT_FILE = "prompt.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 50
LR = 5e-5
MAX_TOKEN_LENGTH = 77
IMAGE_SIZE = 512

# === 加载 ControlNet 和 Pipeline ===
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float32
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)

# === 注入 LoRA 注意力处理器 ===新
unet_lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    init_lora_weights="gaussian",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # 官方推荐组合
)
controlnet_lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    init_lora_weights="gaussian",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)

pipe.unet.add_adapter(unet_lora_config)
pipe.controlnet.add_adapter(controlnet_lora_config)     # 新

# === 确保 LoRA 参数用 float32 精度训练（防止混合精度引发不稳定） ===
cast_training_params(pipe.unet, dtype=torch.float32)
cast_training_params(pipe.controlnet, dtype=torch.float32)

# 冻结所有参数 新
pipe.unet.requires_grad_(False)
pipe.controlnet.requires_grad_(False)
pipe.text_encoder.requires_grad_(True)
pipe.text_encoder.train()

# === 收集参数并创建优化器 ===
trainable_params = []
for proc in pipe.unet.attn_processors.values():   # 新
    for p in proc.parameters():
        if p.requires_grad:
            trainable_params.append(p)
for proc in pipe.controlnet.attn_processors.values():
    for p in proc.parameters():
        if p.requires_grad:
            trainable_params.append(p)
trainable_params += list(pipe.text_encoder.parameters())
# 参数检查
print("✅ 参数检查：")
for name, param in pipe.unet.named_parameters():
    if param.requires_grad:
        print(f"[UNet] 训练参数: {name} - {param.shape}")
for name, param in pipe.controlnet.named_parameters():
    if param.requires_grad:
        print(f"[ControlNet] 训练参数: {name} - {param.shape}")
for name, param in pipe.text_encoder.named_parameters():
    if param.requires_grad:
        print(f"[TextEncoder] 训练参数: {name} - {param.shape}")
optimizer = torch.optim.AdamW(trainable_params, lr=LR)
trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
print(f"🧮 Optimizer 中可训练参数总数: {trainable_count}")    # 新

pipe.unet.train()
pipe.controlnet.train()

# === 加载 Tokenizer ===
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# === 自定义数据集 ===
class VisDroneControlNetDataset(Dataset):
    def __init__(self, root_dir, prompt_file, tokenizer, max_length=MAX_TOKEN_LENGTH):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(os.path.join(root_dir, prompt_file), "r") as f:
            self.entries = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image_path = os.path.join(self.root_dir, item["image_path"])
        layout_path = os.path.join(self.root_dir, item["layout_path"])
        prompt = item["prompt"]

        image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        layout = Image.open(layout_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        # 使用pipe的feature_extractor处理图像和布局
        image_tensor = pipe.feature_extractor(images=image, return_tensors="pt").pixel_values[0]
        layout_tensor = pipe.feature_extractor(images=layout, return_tensors="pt").pixel_values[0]

        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "layout": layout_tensor,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }

dataset = VisDroneControlNetDataset(DATA_DIR, PROMPT_FILE, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)

# === 尝试加载断点 ===
start_epoch =1
for epoch in range(EPOCHS, 0, -1):
    unet_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}")
    if os.path.exists(unet_path):
        print(f"恢复 epoch {epoch} 的检查点...")
        pipe.unet = pipe.unet.from_pretrained(unet_path).to(DEVICE)
        pipe.controlnet = pipe.controlnet.from_pretrained(os.path.join(CHECKPOINT_DIR, f"controlnet_epoch_{epoch}")).to(DEVICE)
        pipe.text_encoder = pipe.text_encoder.from_pretrained(os.path.join(CHECKPOINT_DIR, f"text_encoder_epoch_{epoch}")).to(DEVICE)
        optimizer.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"optimizer_epoch_{epoch}.pt"), map_location=DEVICE))
        start_epoch = epoch + 1
        break

# === 训练循环 ===
for epoch in range(start_epoch, EPOCHS+1):
    pipe.unet.train()
    pipe.controlnet.train()
    pipe.text_encoder.train()

    loop = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    total_loss = 0
    step_count = 0
    for batch in loop:
        optimizer.zero_grad()

        # 移动数据到设备
        image = batch["image"].to(DEVICE, dtype=torch.float32)
        layout = batch["layout"].to(DEVICE, dtype=torch.float32)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # 编码文本
        encoder_hidden_states = pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0].to(dtype=torch.float32).to(DEVICE)

        # 编码图像至latent
        latents = pipe.vae.encode(image).latent_dist.sample().to(DEVICE)
        latents = latents * pipe.vae.config.scaling_factor
        latents = latents.to(dtype=torch.float32)

        # 采样随机时间步
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()

        # 添加噪声
        noise = torch.randn_like(latents).to(DEVICE)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # ControlNet前向
        controlnet_out = pipe.controlnet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=layout,
            return_dict=True,
        )

        # UNet前向，融合ControlNet输出
        unet_out = pipe.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=controlnet_out.down_block_res_samples,
            mid_block_additional_residual=controlnet_out.mid_block_res_sample,
            return_dict=True,
        )

        # 预测噪声
        noise_pred = unet_out.sample

        # 计算损失
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        for name, param in pipe.unet.named_parameters():  # 新
            if param.requires_grad and param.grad is not None:
                print(f"📈 梯度检查: {name} 的梯度均值: {param.grad.abs().mean().item():.6f}")
                break   # 新
        torch.nn.utils.clip_grad_norm_(list(pipe.unet.parameters()) +
                                       list(pipe.controlnet.parameters()) +
                                       list(pipe.text_encoder.parameters()), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        step_count += 1
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / step_count if step_count > 0 else 0
    print(f"平均 Loss（Epoch {epoch}）: {avg_loss:.6f}")

    if epoch == start_epoch:     # 新
        initial_param_snapshot = pipe.unet.attn_processors[
            "mid_block.attn1.processor"].to_q_lora.lora_A.weight.detach().clone()
    elif epoch > start_epoch:
        current_param = pipe.unet.attn_processors["mid_block.attn1.processor"].to_q_lora.lora_A.weight
        delta = (current_param - initial_param_snapshot).abs().mean().item()
        print(f"🧪 参数变化均值 (mid_block to_q): {delta:.6f}")    # 新

    if epoch == EPOCHS:
        pipe.unet.save_pretrained(os.path.join(OUTPUT_DIR, "unet"))
        pipe.controlnet.save_pretrained(os.path.join(OUTPUT_DIR, "controlnet"))
        pipe.text_encoder.save_pretrained(os.path.join(OUTPUT_DIR, "text_encoder"))
        torch.save(optimizer.state_dict(), os.path.join(OUTPUT_DIR, "optimizer.pt"))
        print("训练完成，最终模型已保存。")
    else:
        pipe.unet.save_pretrained(os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}"))
        pipe.controlnet.save_pretrained(os.path.join(CHECKPOINT_DIR, f"controlnet_epoch_{epoch}"))
        pipe.text_encoder.save_pretrained(os.path.join(CHECKPOINT_DIR, f"text_encoder_epoch_{epoch}"))
        torch.save(optimizer.state_dict(), os.path.join(CHECKPOINT_DIR, f"optimizer_epoch_{epoch}.pt"))
        print(f"保存 epoch {epoch} 的权重完成。")
