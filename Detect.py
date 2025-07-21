import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
import os

# 设置路径
UNET_PATH = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints/unet_epoch_6"
PRETRAINED_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/test_sample/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型组件
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(UNET_PATH).to(device)
scheduler = DDIMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler")

# 编写 prompt
prompt = "There are 5 pedestrians, 5 peoples, 5 cars, 5 vans, 5 trucks, and 5 awning-tricycle in the image."
max_length = tokenizer.model_max_length
text_inputs = tokenizer(prompt, padding="max_length", max_length=max_length, return_tensors="pt")
input_ids = text_inputs.input_ids.to(device)

# 获取文本特征
with torch.no_grad():
    encoder_hidden_states = text_encoder(input_ids)[0]

# 采样准备
height, width = 512, 512
latents = torch.randn((1, unet.config.in_channels, height // 8, width // 8), device=device)
scheduler.set_timesteps(50)
latents = latents * scheduler.init_noise_sigma

# 逐步反推
for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = unet(latents, t, encoder_hidden_states).sample

    latents = scheduler.step(noise_pred, t, latents).prev_sample

# 解码图像
with torch.no_grad():
    image = vae.decode(latents / vae.config.scaling_factor).sample

# 后处理
image = (image / 2 + 0.5).clamp(0, 1)
image = image[0].cpu().permute(1, 2, 0).numpy()
image = (image * 255).astype(np.uint8)
Image.fromarray(image).convert("RGB").save(os.path.join(OUTPUT_DIR, "result.jpg"), format="JPEG")
print("✅ 图像生成完成，保存到：", os.path.join(OUTPUT_DIR, "result.jpg"))
