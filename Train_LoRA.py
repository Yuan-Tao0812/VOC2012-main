import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch import nn, optim
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.models.attention_processor import LoRAAttnProcessor


# === 配置 ===
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/"
PROMPT_FILE = "prompt.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-4
MAX_TOKEN_LENGTH = 77

# === 加载模型 ===
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",  # 或 "runwayml/stable-diffusion-v1-5"
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(DEVICE)
pipe.enable_model_cpu_offload()  # 节省显存

# === 注入 LoRA（diffusers 自带）
pipe.unet.set_attn_processor(LoRAAttnProcessor())  # unet注入LoRA模块
pipe.controlnet.set_attn_processor(LoRAAttnProcessor())  # controlnet注入LoRA模块

# 设置可训练参数
for module in [pipe.unet, pipe.controlnet]:
    for name, submodule in module.named_modules():
        if isinstance(submodule, LoRAAttnProcessor):
            for param in submodule.parameters():
                param.requires_grad = True

pipe.unet.train()
pipe.controlnet.train()

# === 文本编码 ===
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = pipe.text_encoder
text_encoder.train()

# === 数据集定义 ===
class ControlnetDataset(Dataset):
    def __init__(self, root_dir, jsonl_file, tokenizer, max_length=MAX_TOKEN_LENGTH):
        self.root = root_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(os.path.join(root_dir, jsonl_file), "r") as f:
            self.entries = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image_path = os.path.join(self.root, item["image_path"])
        layout_path = os.path.join(self.root, item["layout_path"])
        prompt = item["prompt"]

        image = Image.open(image_path).convert("RGB").resize((512, 512))
        layout = Image.open(layout_path).convert("RGB").resize((512, 512))

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
            "prompt": prompt,
        }

dataset = ControlnetDataset(DATA_DIR, PROMPT_FILE, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === 优化器（只训练 LoRA 和文本编码器）
optimizer = optim.AdamW(
    list(filter(lambda p: p.requires_grad, pipe.unet.parameters())) +
    list(filter(lambda p: p.requires_grad, pipe.controlnet.parameters())) +
    list(text_encoder.parameters()),
    lr=LR
)

# === 训练主循环 ===
for epoch in range(EPOCHS):
    pipe.unet.train()
    pipe.controlnet.train()
    text_encoder.train()

    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch in loop:
        optimizer.zero_grad()

        image_tensors = batch["image"].to(DEVICE)
        layout_tensors = batch["layout"].to(DEVICE)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        noise = torch.randn_like(image_tensors).to(DEVICE)
        timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (BATCH_SIZE,), device=DEVICE).long()

        outputs = pipe(
            image=image_tensors,
            timesteps=timesteps,
            prompt_embeds=encoder_hidden_states,
            controlnet_conditioning_image=layout_tensors,
            return_dict=True,
        )

        noise_pred = outputs.sample

        loss = nn.MSELoss()(noise_pred, noise)
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


print("训练结束，模型已保存。")
pipe.unet.save_attn_procs(os.path.join(OUTPUT_DIR, "unet_lora"))
pipe.controlnet.save_attn_procs(os.path.join(OUTPUT_DIR, "controlnet_lora"))
text_encoder.save_pretrained(os.path.join(OUTPUT_DIR, "text_encoder"))
pipe.unet.save_pretrained(os.path.join(OUTPUT_DIR, "unet_full"), safe_serialization=True)
print("所有模型保存完毕。")