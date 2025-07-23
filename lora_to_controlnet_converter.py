"""
将训练好的LoRA权重合并到SD模型中，并保存为ControlNet训练所需的格式
"""

import torch
import os
from diffusers import StableDiffusionPipeline
from safetensors.torch import save_file, load_file


def merge_lora_to_full_model(
        base_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
        lora_weights_path="/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/",
        output_path="/content/drive/MyDrive/VisDrone2019-YOLO/sd15_with_lora.ckpt",
        lora_scale=1.0
):
    """
    将LoRA权重合并到完整的SD模型中

    Args:
        base_model_path: 基础SD模型路径
        lora_weights_path: LoRA权重目录
        output_path: 输出的完整模型路径
        lora_scale: LoRA权重缩放系数
    """

    print("🔄 开始合并LoRA权重到完整模型...")

    # 1. 加载基础模型
    print("📥 加载基础SD模型...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # 使用float32确保精度
        safety_checker=None,
        requires_safety_checker=False
    )

    # 2. 加载LoRA权重
    print("📥 加载LoRA权重...")
    pipeline.load_lora_weights(lora_weights_path)

    # 3. 设置LoRA缩放系数
    if hasattr(pipeline, 'set_adapters'):
        pipeline.set_adapters(["default"], [lora_scale])

    # 4. 合并LoRA权重到UNet
    print("🔗 合并LoRA权重...")
    pipeline.fuse_lora(lora_scale=lora_scale)

    # 5. 构建完整的state_dict
    print("💾 构建完整模型state_dict...")

    # 获取各个组件的state_dict
    unet_state_dict = pipeline.unet.state_dict()
    vae_state_dict = pipeline.vae.state_dict()
    text_encoder_state_dict = pipeline.text_encoder.state_dict()

    # 构建完整的模型state_dict（ControlNet格式）
    full_state_dict = {}

    # UNet权重 - 添加"model.diffusion_model."前缀
    for key, value in unet_state_dict.items():
        new_key = f"model.diffusion_model.{key}"
        full_state_dict[new_key] = value

    # VAE权重
    # Decoder
    for key, value in vae_state_dict.items():
        if key.startswith("decoder."):
            new_key = f"first_stage_model.{key}"
            full_state_dict[new_key] = value
        elif key.startswith("encoder."):
            new_key = f"first_stage_model.{key}"
            full_state_dict[new_key] = value
        elif key.startswith("quant_conv."):
            new_key = f"first_stage_model.{key}"
            full_state_dict[new_key] = value
        elif key.startswith("post_quant_conv."):
            new_key = f"first_stage_model.{key}"
            full_state_dict[new_key] = value

    # Text Encoder权重
    for key, value in text_encoder_state_dict.items():
        new_key = f"cond_stage_model.transformer.{key}"
        full_state_dict[new_key] = value

    # 6. 保存为.ckpt格式
    print(f"💾 保存完整模型到: {output_path}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为.ckpt格式
    torch.save({
        'state_dict': full_state_dict,
        'global_step': 0,
        'epoch': 0,
    }, output_path)

    print("✅ 合并完成！")

    # 7. 验证保存的模型
    print("🔍 验证保存的模型...")
    try:
        checkpoint = torch.load(output_path, map_location="cpu")
        state_dict = checkpoint['state_dict']

        print(f"📊 模型统计:")
        print(f"  - 总参数数量: {len(state_dict)}")
        print(f"  - UNet参数: {len([k for k in state_dict.keys() if 'diffusion_model' in k])}")
        print(f"  - VAE参数: {len([k for k in state_dict.keys() if 'first_stage_model' in k])}")
        print(f"  - TextEncoder参数: {len([k for k in state_dict.keys() if 'cond_stage_model' in k])}")
        print(f"  - 文件大小: {os.path.getsize(output_path) / 1024 / 1024 / 1024:.2f} GB")

        print("✅ 模型验证通过！")

    except Exception as e:
        print(f"❌ 模型验证失败: {e}")

    # 8. 清理内存
    del pipeline
    torch.cuda.empty_cache()

    return output_path


def create_controlnet_config():
    """
    创建ControlNet训练所需的配置文件
    """
    config_content = """
# ControlNet配置文件
model:
  base_learning_rate: 1.0e-5
  target: cldm.cldm.ControlLDM
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    control_key: "hint"
    image_size: 64
    channels: 4
    cond_stage_trainable: false
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False
    only_mid_control: False

    control_stage_config:
      target: cldm.controlnet.ControlNet
      params:
        image_size: 32 # unused
        in_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    unet_config:
      target: cldm.cldm.ControlledUnetModel
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_channels: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
"""

    return config_content


# 使用示例
if __name__ == "__main__":
    # 转换LoRA模型
    merged_model_path = merge_lora_to_full_model(
        base_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
        lora_weights_path="/content/drive/MyDrive/VisDrone2019-YOLO/trained_lora_controlnet/",
        output_path="/content/drive/MyDrive/VisDrone2019-YOLO/sd15_with_lora.ckpt",
        lora_scale=1.0
    )

    # 保存配置文件
    config_content = create_controlnet_config()
    config_path = "/content/drive/MyDrive/VisDrone2019-YOLO/controlnet_config.yaml"

    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"\n🎯 转换完成！现在你可以用于ControlNet训练:")
    print(f"📁 合并后的模型: {merged_model_path}")
    print(f"📁 配置文件: {config_path}")

    print(f"\n🔧 ControlNet训练代码修改:")
    print(f"resume_path = '{merged_model_path}'")
    print(f"# 将 './models/control_sd15_ini.ckpt' 改为上面的路径")