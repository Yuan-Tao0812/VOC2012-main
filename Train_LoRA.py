import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch import nn, optim
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from peft import LoraConfig, get_peft_model

# === 配置 ===
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/"
PROMPT_FILE = "prompt.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 30
LR = 1e-4
MAX_TOKEN_LENGTH = 77

# === 加载模型 ===
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(DEVICE)
pipe.enable_model_cpu_offload()  # 节省显存

# === LoRA 配置 ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type="TEXT_TO_IMAGE"
)

# 对 UNet 和 ControlNet 注入 LoRA
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.controlnet = get_peft_model(pipe.controlnet, lora_config)

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

        # 读图并resize
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        layout = Image.open(layout_path).convert("RGB").resize((512, 512))

        # 文本tokenize
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "layout": layout,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "prompt": prompt,
        }


dataset = ControlnetDataset(DATA_DIR, PROMPT_FILE, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === 优化器 ===
optimizer = optim.AdamW(
    list(pipe.unet.parameters()) + list(pipe.controlnet.parameters()) + list(text_encoder.parameters()),
    lr=LR
)

# === 训练主循环 ===
from diffusers.utils import PIL_INTERPOLATION

for epoch in range(EPOCHS):
    pipe.unet.train()
    pipe.controlnet.train()
    text_encoder.train()

    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch in loop:
        optimizer.zero_grad()

        # 处理图像和layout
        images = [img for img in batch["image"]]
        layouts = [lay for lay in batch["layout"]]

        # 使用 pipeline 自带的 feature extractor 将 PIL 图转 tensor
        image_tensors = pipe.feature_extractor(images=images, return_tensors="pt").pixel_values.to(DEVICE)
        layout_tensors = pipe.feature_extractor(images=layouts, return_tensors="pt").pixel_values.to(DEVICE)

        # 编码文本
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        # 计算噪声和时间步
        noise = torch.randn_like(image_tensors).to(DEVICE)
        timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (BATCH_SIZE,), device=DEVICE).long()

        # 得到噪声的预测
        noise_pred = pipe.unet(image_tensors, timesteps, encoder_hidden_states, controlnet_cond=layout_tensors).sample

        # 计算loss（MSE）
        loss = nn.MSELoss()(noise_pred, noise)
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    # 每隔几个epoch保存一次模型
    if (epoch + 1) % 5 == 0:
        print(f"Saving model at epoch {epoch + 1} ...")
        pipe.unet.save_pretrained(os.path.join(OUTPUT_DIR, f"unet_lora_epoch{epoch + 1}"))
        pipe.controlnet.save_pretrained(os.path.join(OUTPUT_DIR, f"controlnet_lora_epoch{epoch + 1}"))
        text_encoder.save_pretrained(os.path.join(OUTPUT_DIR, f"text_encoder_epoch{epoch + 1}"))

print("训练结束，模型已保存。")
