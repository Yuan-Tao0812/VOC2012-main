import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# ====== 加载 Stable Diffusion 模型 ======
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16
).to("cuda")

# ====== 加载训练好的 LoRA 权重 ======
LORA_PATH = "/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora/"
pipe.load_lora_weights(LORA_PATH)

# ====== 输入你的 prompt ======
prompt = "2 pedestrians, 3 cars, visdrone style"

# ====== 推理生成图像 ======
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("output.png")
image.show()
