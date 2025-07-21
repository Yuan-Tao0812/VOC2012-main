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
)
from diffusers.training_utils import cast_training_params
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig

DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-renamed/"
PROMPT_FILE = "prompt.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"
CHECKPOINT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints"
PRETRAINED_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
MAX_TOKEN_LENGTH = 77
IMAGE_SIZE = 512
epoch_losses = []
weight_dtype = torch.float32

noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="unet")

unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

unet_lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    init_lora_weights="gaussian",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)

text_encoder.to(DEVICE, dtype=weight_dtype)
unet.to(DEVICE, dtype=weight_dtype)
vae.to(DEVICE, dtype=weight_dtype)

unet.add_adapter(unet_lora_config)

cast_training_params(unet, dtype=torch.float32)

# 过滤出 LoRA 可训练参数
lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))

# 验证：打印可训练参数数量，确认不为0
print(f"可训练参数数量（LoRA）：{len(lora_layers)}")
assert len(lora_layers) > 0, "没有找到任何可训练参数，请检查LoRA是否正确添加"

# 额外打印几个参数名和requires_grad状态，排查异常
print("部分LoRA可训练参数名字和requires_grad状态示例：")
for name, param in unet.named_parameters():
    if param.requires_grad:
        print(f"  {name}: requires_grad={param.requires_grad}")
        break

unet.enable_gradient_checkpointing()

torch.backends.cuda.matmul.allow_tf32 = True

optimizer_cls = torch.optim.AdamW
optimizer = optimizer_cls(
    lora_layers,
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # 归一化为 [-1, 1]
])


class VisDroneControlNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, prompt_file, tokenizer, max_length=MAX_TOKEN_LENGTH, transform=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        with open(os.path.join(root_dir, prompt_file), "r") as f:
            self.entries = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image = Image.open(os.path.join(self.root_dir, item["image"])).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tokenized = self.tokenizer(
            item["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }

dataset = VisDroneControlNetDataset(DATA_DIR, PROMPT_FILE, tokenizer, transform=transform)

def collate_fn(examples):
    pixel_values = torch.stack([ex["image"] for ex in examples])
    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    attention_mask = torch.stack([ex["attention_mask"] for ex in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

num_update_steps_per_epoch = len(dataloader)
max_train_steps = EPOCHS * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=max_train_steps,
)

# 断点恢复逻辑，恢复后确保模型移动和 dtype 设置一致
start_epoch = 1
for epoch in range(EPOCHS, 0, -1):
    unet_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}")
    optimizer_path = os.path.join(CHECKPOINT_DIR, f"optimizer_epoch_{epoch}.pt")
    if os.path.exists(unet_path):
        print(f"恢复 epoch {epoch} 的检查点...")
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        unet.add_adapter(unet_lora_config)
        unet = unet.to_empty(DEVICE)
        unet = unet.to(dtype=weight_dtype)
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
        start_epoch = epoch + 1
        break

accelerator = Accelerator()
unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

# 记录某个LoRA参数训练前后的均值，验证参数是否发生变化
def get_sample_lora_param_mean(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name.lower():
            return param.mean().item()
    return None

for epoch in range(start_epoch, EPOCHS + 1):
    unet.train()
    total_loss = 0.0

    # 训练开始前打印LoRA参数均值
    param_mean_before = get_sample_lora_param_mean(unet)
    print(f"Epoch {epoch} 开始，示例LoRA参数均值: {param_mean_before}")

    loop = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch in loop:
        optimizer.zero_grad()

        image = batch["pixel_values"].to(DEVICE, dtype=weight_dtype)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_layers, 1.0)
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loop)
    epoch_losses.append(avg_loss)
    print(f"平均 Loss（Epoch {epoch}）: {avg_loss:.6f}")

    # 训练结束后打印LoRA参数均值，检查是否变化
    param_mean_after = get_sample_lora_param_mean(unet)
    print(f"Epoch {epoch} 结束，示例LoRA参数均值: {param_mean_after}")
    if param_mean_before == param_mean_after:
        print(f"警告：Epoch {epoch} LoRA参数均值无变化，训练可能未生效！")

    if epoch == EPOCHS:
        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(os.path.join(OUTPUT_DIR, "unet"))
        torch.save(optimizer.state_dict(), os.path.join(OUTPUT_DIR, "optimizer.pt"))
    else:
        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}"))
        torch.save(optimizer.state_dict(), os.path.join(CHECKPOINT_DIR, f"optimizer_epoch_{epoch}.pt"))

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label="Avg Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()
