import os
import json
from PIL import Image
from tqdm import tqdm
import torch

from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from peft.trainer import PeftTrainer
from torch.utils.data import Dataset

# ====== 数据路径配置 ======
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 加载 prompt.jsonl 数据集 ======
class PromptDataset(Dataset):
    def __init__(self, data_dir, jsonl_file, feature_extractor):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, jsonl_file), 'r') as f:
            lines = f.readlines()
        self.items = [json.loads(line.strip()) for line in lines]
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = os.path.join(self.data_dir, item["image_path"])
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]
        return {
            "pixel_values": pixel_values,
            "prompt": item["prompt"]
        }

# ====== 加载 Stable Diffusion 模型 ======
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# ====== 构建数据集 ======
dataset = PromptDataset(DATA_DIR, "prompt.jsonl", pipe.feature_extractor)

# ====== 配置 LoRA 参数 ======
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 兼容 PEFT 的模块名
    lora_dropout=0.1,
    bias="none",
    task_type="TEXT_TO_IMAGE"
)

# ====== 训练参数 ======
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

# ====== 启动训练 LoRA（只微调 UNet） ======
trainer = PeftTrainer(
    model=pipe.unet,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)