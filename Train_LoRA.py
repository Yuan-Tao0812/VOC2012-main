import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.models.attention_processor import LoRAAttnProcessor

# === 配置参数 ===
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/"
PROMPT_FILE = "prompt.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-4
MAX_TOKEN_LENGTH = 77
IMAGE_SIZE = 512

# === 加载 ControlNet 和 Pipeline ===
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.enable_model_cpu_offload()

# === 注入 LoRA 注意力处理器 ===
def inject_lora(model):
    attn_processors = {
        name: LoRAAttnProcessor() for name in model.attn_processors.keys()
    }
    model.set_attn_processor(attn_processors)

inject_lora(pipe.unet)
inject_lora(pipe.controlnet)

# 设置可训练参数，只训练 LoRA 层和文本编码器
for param in pipe.unet.parameters():
    param.requires_grad = False
for param in pipe.controlnet.parameters():
    param.requires_grad = False


pipe.text_encoder.train()
for param in pipe.text_encoder.parameters():
    param.requires_grad = True

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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === 优化器（只训练 LoRA 和 text_encoder） ===
def get_lora_parameters(attn_processors):
    params = []
    for proc in attn_processors.values():
        if isinstance(proc, LoRAAttnProcessor):
            params.extend(proc.parameters())
    return params

optimizer = torch.optim.AdamW(
    list(pipe.unet.get_attn_procs().parameters()) +
    list(pipe.controlnet.get_attn_procs().parameters()) +
    list(pipe.text_encoder.parameters()),
    lr=LR,
)

# === 训练循环 ===
for epoch in range(EPOCHS):
    pipe.unet.train()
    pipe.controlnet.train()
    pipe.text_encoder.train()

    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch in loop:
        optimizer.zero_grad()

        # 移动数据到设备
        image = batch["image"].to(DEVICE, dtype=torch.float16)
        layout = batch["layout"].to(DEVICE, dtype=torch.float16)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # 编码文本
        encoder_hidden_states = pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0].to(dtype=torch.float16)

        # 编码图像至latent
        latents = pipe.vae.encode(image.unsqueeze(0)).latent_dist.sample()  # [1,C,H,W]
        latents = latents * pipe.vae.config.scaling_factor
        latents = latents.to(dtype=torch.float16)

        # 采样随机时间步
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()

        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # ControlNet 条件输入
        layout = layout.unsqueeze(0).to(dtype=torch.float16)

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
        optimizer.step()

        loop.set_postfix(loss=loss.item())

# 保存模型LoRA权重和文本编码器
pipe.unet.save_attn_procs(os.path.join(OUTPUT_DIR, "unet_lora"))
pipe.controlnet.save_attn_procs(os.path.join(OUTPUT_DIR, "controlnet_lora"))
pipe.text_encoder.save_pretrained(os.path.join(OUTPUT_DIR, "text_encoder"))

print("训练完成，模型保存完毕。")
