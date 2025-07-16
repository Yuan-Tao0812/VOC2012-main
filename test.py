import os
import json
import random
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# === 路径设置 ===
BASE_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/test_sample"
os.makedirs(BASE_DIR, exist_ok=True)
IMAGE_PATH = os.path.join(BASE_DIR, "test.jpg")
LAYOUT_PATH = os.path.join(BASE_DIR, "test.png")
LABEL_PATH = os.path.join(BASE_DIR, "test.txt")
PROMPT_JSONL = os.path.join(BASE_DIR, "prompt1.jsonl")

# === 模型路径 ===
UNET_PATH = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints/unet_epoch_32"
CONTROLNET_PATH = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints/controlnet_epoch_32"
TEXT_ENCODER_PATH = "/content/drive/MyDrive/VisDrone2019-YOLO/checkpoints/text_encoder_epoch_32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512

# === 类别名称与映射（YOLO 格式类别 ID）===
LABEL_MAP = {
    "pedestrians": 0,
    "peoples": 1,
    "cars": 2,
    "vans": 3,
    "trucks": 4,
    "tricycles": 5,
    "motors": 6
}

# === 解析 prompt 中的对象数量 ===
def parse_prompt(prompt):
    items = prompt.split(",")
    objs = []
    for item in items:
        parts = item.strip().split()
        if len(parts) < 2: continue
        count, name = parts[0], parts[1]
        try:
            n = int(count)
            key = name.lower()
            if key in LABEL_MAP:
                objs.extend([LABEL_MAP[key]] * n)
        except:
            continue
    return objs  # List[int]: 类别 ID 列表

# === YOLO标签写入 ===
def generate_labels(objs, width, height, path):
    with open(path, "w") as f:
        for cls in objs:
            # 随机中心和宽高
            xc = random.uniform(0.1, 0.9)
            yc = random.uniform(0.1, 0.9)
            w = random.uniform(0.05, 0.15)
            h = random.uniform(0.05, 0.15)
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# === layout图生成 ===
def generate_layout(objs, size=512):
    img = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(img)
    for _ in objs:
        x0 = random.randint(0, size)
        y0 = random.randint(0, size)
        x1 = x0 + random.randint(10, 50)
        y1 = y0 + random.randint(10, 50)
        draw.rectangle((x0, y0, x1, y1), outline="black", width=2)
    return img

# === 加载训练模型 ===
def load_pipe():
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH).to(DEVICE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
    )
    pipe.unet = pipe.unet.from_pretrained(UNET_PATH).to(DEVICE)
    pipe.text_encoder = pipe.text_encoder.from_pretrained(TEXT_ENCODER_PATH).to(DEVICE)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(DEVICE)

# === 主过程 ===
def main():
    prompt = "10 pedestrians, 27 peoples, 32 cars, 11 vans, 9 trucks, 2 tricycles, 34 motors, visdrone style"
    objs = parse_prompt(prompt)

    print(f"从 prompt 中解析出 {len(objs)} 个目标用于标签生成")

    layout = generate_layout(objs, size=IMAGE_SIZE)
    layout.save(LAYOUT_PATH)

    pipe = load_pipe()
    result = pipe(prompt, image=layout, num_inference_steps=30)
    result.images[0].save(IMAGE_PATH)

    generate_labels(objs, IMAGE_SIZE, IMAGE_SIZE, LABEL_PATH)

    with open(PROMPT_JSONL, "w") as f:
        json.dump({
            "image_path": os.path.basename(IMAGE_PATH),
            "layout_path": os.path.basename(LAYOUT_PATH),
            "prompt": prompt
        }, f)
        f.write("\n")

    print("✅ 所有文件保存完成：")
    print("🖼 图像:", IMAGE_PATH)
    print("📐 Layout:", LAYOUT_PATH)
    print("🏷 标签:", LABEL_PATH)
    print("📝 prompt1.jsonl:", PROMPT_JSONL)

if __name__ == "__main__":
    main()
