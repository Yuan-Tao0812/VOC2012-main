# 安装依赖（Colab执行一次）
!pip install diffusers transformers accelerate datasets peft huggingface_hub --quiet
!pip install --upgrade Pillow tqdm --quiet

import os
import json
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer, TrainingArguments
from peft import LoraConfig
from peft.trainer import PeftTrainer

# ==== 路径配置 ====
DATA_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/"
PROMPT_FILE = "prompt.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 加载模型 ====
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_scribble",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",  # 推荐使用这个官方v1.5模型
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# ==== 自定义数据集 ====
class ControlnetDataset(Dataset):
    def __init__(self, root_dir, jsonl_file, feature_extractor, tokenizer):
        self.root = root_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        with open(os.path.join(root_dir, jsonl_file), 'r') as f:
            self.entries = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image_path = os.path.join(self.root, item["image_path"])
        layout_path = os.path.join(self.root, item["layout_path"])
        prompt = item["prompt"]

        image = Image.open(image_path).convert("RGB").resize((512, 512))
        layout = Image.open(layout_path).convert("RGB").resize((512, 512))

        image_tensor = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]
        layout_tensor = self.feature_extractor(images=layout, return_tensors="pt").pixel_values[0]

        encoding = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        return {
            "pixel_values": image_tensor,
            "controlnet_cond": layout_tensor,
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0],
        }

dataset = ControlnetDataset(DATA_DIR, PROMPT_FILE, pipe.feature_extractor, tokenizer)

# ==== 配置 LoRA ====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # 对UNet中的Cross-Attention层微调
    lora_dropout=0.1,
    bias="none",
    task_type="TEXT_TO_IMAGE"
)

# LoRA只套pipe.unet，ControlNet参数冻结
from peft import get_peft_model
model = get_peft_model(pipe.unet, lora_config)
model.train()

# ==== 训练参数 ====
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=50,
    learning_rate=1e-4,
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none"
)

# ==== 自定义 Trainer (兼容 ControlNet输入) ====
from transformers import Trainer

class ControlNetTrainer(PeftTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        pixel_values = inputs.pop("pixel_values")
        controlnet_cond = inputs.pop("controlnet_cond")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")

        # forward with controlnet_cond和prompt编码
        outputs = model(
            pixel_values=pixel_values.unsqueeze(0).to(model.device),
            controlnet_cond=controlnet_cond.unsqueeze(0).to(model.device),
            input_ids=input_ids.unsqueeze(0).to(model.device),
            attention_mask=attention_mask.unsqueeze(0).to(model.device),
        )
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

trainer = ControlNetTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

# ==== 开始训练 ====
trainer.train()
trainer.save_model(OUTPUT_DIR)

print("训练完成，模型保存在：", OUTPUT_DIR)
