"""
Gradio工作台配置文件
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent

# 模型路径配置
MODEL_DIRS = {
    "stable_diffusion": PROJECT_ROOT / "Stable_diffusion",
    "dalle_mini": PROJECT_ROOT / "DALLE_mini", 
    "blip_video": PROJECT_ROOT / "BLIP_video",
    "nano_gpt": PROJECT_ROOT / "nano_gpt"
}

# 本地模型目录
LOCAL_MODEL_DIRS = {
    "stable_diffusion": BASE_DIR / "models" / "stable_diffusion",
    "blip_video": BASE_DIR / "models" / "blip_video",
    "nano_gpt": BASE_DIR / "models" / "nano_gpt",
    "dalle_mini": BASE_DIR / "models" / "dalle_mini"
}

# 创建本地模型目录
for model_dir in LOCAL_MODEL_DIRS.values():
    (model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (model_dir / "cache").mkdir(parents=True, exist_ok=True)
    (model_dir / "configs").mkdir(parents=True, exist_ok=True)

# 输出路径配置
OUTPUT_DIRS = {
    "images": BASE_DIR / "outputs" / "images",
    "videos": BASE_DIR / "outputs" / "videos", 
    "text": BASE_DIR / "outputs" / "text",
    "cache": BASE_DIR / "cache"
}

# 创建输出目录
for dir_path in OUTPUT_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_CONFIGS = {
    "stable_diffusion": {
        "default_model": "runwayml/stable-diffusion-v1-5",
        "sdxl_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "controlnet_models": {
            "pose": "lllyasviel/sd-controlnet-openpose",
            "depth": "lllyasviel/sd-controlnet-depth",
            "canny": "lllyasviel/sd-controlnet-canny"
        },
        "local_models": {
            "sd-v1-5": LOCAL_MODEL_DIRS["stable_diffusion"] / "checkpoints" / "stable-diffusion-v1-5",
            "sdxl": LOCAL_MODEL_DIRS["stable_diffusion"] / "checkpoints" / "sdxl",
            "custom": LOCAL_MODEL_DIRS["stable_diffusion"] / "checkpoints" / "custom"
        }
    },
    "dalle_mini": {
        "default_model": "dalle-mini/dalle-mini",
        "local_models": {
            "dalle-mini": LOCAL_MODEL_DIRS["dalle_mini"] / "checkpoints" / "dalle-mini"
        }
    },
    "blip_video": {
        "default_model": "Salesforce/blip-image-captioning-base",
        "local_models": {
            "blip-video": LOCAL_MODEL_DIRS["blip_video"] / "checkpoints" / "blip-video-captioning-base" / "models--Salesforce--blip-image-captioning-base" / "snapshots" / "82a37760796d32b1411fe092ab5d4e227313294b"
        }
    },
    "nano_gpt": {
        "default_model": "gpt2",
        "local_models": {
            "gpt2": LOCAL_MODEL_DIRS["nano_gpt"] / "checkpoints" / "gpt2",
            "custom": LOCAL_MODEL_DIRS["nano_gpt"] / "checkpoints" / "custom"
        }
    }
}

# 设备配置
DEVICE_CONFIG = {
    "auto_detect": True,
    "prefer_mps": True,  # Mac M系列优先使用MPS
    "fallback_cpu": True,
    "memory_efficient": True
}

# API配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 7860,
    "workers": 1,
    "max_upload_size": 100 * 1024 * 1024,  # 100MB
    "timeout": 300  # 5分钟
}

# 队列配置
QUEUE_CONFIG = {
    "max_size": 10,
    "timeout": 600,  # 10分钟
    "retry_attempts": 3
}

# 缓存配置
CACHE_CONFIG = {
    "enable": True,
    "max_size": 1000,
    "ttl": 3600  # 1小时
}

# 监控配置
MONITORING_CONFIG = {
    "enable": True,
    "metrics_port": 9090,
    "log_level": "INFO"
}

# 安全配置
SECURITY_CONFIG = {
    "enable_nsfw_filter": True,
    "enable_watermark": True,
    "max_prompt_length": 500,
    "blocked_keywords": [
        "explicit", "nsfw", "adult", "violence", "hate"
    ]
}

# 默认参数
DEFAULT_PARAMS = {
    "text2img": {
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "num_images_per_prompt": 1,
        "seed": None
    },
    "img2img": {
        "strength": 0.7,
        "guidance_scale": 7.5,
        "num_inference_steps": 20,
        "num_images_per_prompt": 1,
        "seed": None
    },
    "video_caption": {
        "max_length": 100,
        "num_beams": 5,
        "temperature": 0.7
    }
}

# 语言配置
LANGUAGE_CONFIG = {
    "supported_languages": [
        "en", "zh", "ja", "ko", "es", "fr", "de", "it", "ru", "ar"
    ],
    "default_language": "en",
    "auto_translate": True
}

# 导出配置
EXPORT_CONFIG = {
    "video_formats": ["mp4", "avi", "mov"],
    "subtitle_formats": ["srt", "vtt", "ass"],
    "image_formats": ["png", "jpg", "jpeg", "webp"],
    "text_formats": ["txt", "json", "csv"]
}
