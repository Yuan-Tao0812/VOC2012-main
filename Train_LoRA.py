import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.utils import load_image, make_image_grid
from peft import get_peft_model, LoraConfig
from peft.trainer import PeftTrainer


# ==== 1. 数据集定义 ====
class ControlNetDataset(Dataset):
    def __init__(self, jsonl_path, root_dir, feature_extractor):
        with open(jsonl_path, 'r') as f:
            self.entries = [json.loads(line.strip()) for line in f]
        self.root = root_dir
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image_path = os.path.join(self.root, item["image_path"])
        control_path = os.path.join(self.root, item["layout_path"])
        prompt = item["prompt"]

        image = Image.open(image_path).convert("RGB").resize((512, 512))
        control = Image.open(control_path).convert("RGB").resize((512, 512))

        image_tensor = self.feature_extractor(image, return_tensors="pt").pixel_values[0]
        control_tensor = self.feature_extractor(control, return_tensors="pt").pixel_values[0]

        return {
            "pixel_values": image_tensor,
            "control_image": control_tensor,
            "prompt": prompt
        }


# ==== 2. 路径配置 ====
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora/"
JSONL_PATH = os.path.join(DATA_DIR, "prompt.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==== 3. 加载模型（Stable Diffusion + ControlNet 空模型） ====
base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_unet(Unet2DConditionModel.from_pretrained(base_model_id, subfolder="unet"))
vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16
).to("cuda")


# ==== 4. 构建数据集 ====
dataset = ControlNetDataset(JSONL_PATH, DATA_DIR, pipe.feature_extractor)


# ==== 5. LoRA 参数 ====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # 适配 UNet + ControlNet
    task_type="TEXT_TO_IMAGE"
)


# ==== 6. 训练参数 ====
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=100,
    learning_rate=1e-4,
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none"
)


# ==== 7. 构建 LoRA Trainer ====
trainer = PeftTrainer(
    model=pipe.unet,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
