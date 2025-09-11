"""
Stable Diffusion模块 - 集成文生图、图生图、ControlNet等功能
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any
import logging
from pathlib import Path

from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler
)
from diffusers.utils import logging as diffusers_logging

from utils.device_manager import device_manager
from configs.settings import MODEL_CONFIGS, DEFAULT_PARAMS, OUTPUT_DIRS

logger = logging.getLogger(__name__)

# 设置diffusers日志级别
diffusers_logging.set_verbosity_error()


class StableDiffusionModule:
    """Stable Diffusion模块"""
    
    def __init__(self):
        self.device = device_manager.get_device()
        self.torch_dtype = device_manager.get_torch_dtype()
        self.pipelines = {}
        self.controlnet_models = {}
        self._load_models()
    
    def _load_models(self):
        """加载模型"""
        try:
            logger.info("Loading Stable Diffusion models...")
            
            # 加载基础SD模型
            self._load_sd_pipeline()
            
            # 暂时禁用SDXL模型加载
            # self._load_sdxl_pipeline()
            
            # 暂时禁用ControlNet模型加载
            # self._load_controlnet_models()
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_sd_pipeline(self):
        """加载基础SD管道"""
        # 优先使用本地模型
        local_model_path = MODEL_CONFIGS["stable_diffusion"]["local_models"]["sd-v1-5"]
        if local_model_path.exists():
            model_id = str(local_model_path)
            logger.info(f"Using local SD model: {model_id}")
        else:
            model_id = MODEL_CONFIGS["stable_diffusion"]["default_model"]
            logger.info(f"Using remote SD model: {model_id}")
        
        # 检查是否是单个safetensors文件
        single_file = Path(model_id) / "v1-5-pruned.safetensors"
        if single_file.exists():
            # 使用单个文件加载
            self.pipelines["text2img"] = StableDiffusionPipeline.from_single_file(
                str(single_file),
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            # 使用标准方式加载
            self.pipelines["text2img"] = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
        
        # 为img2img创建相同的pipeline（因为from_single_file不支持img2img）
        if single_file.exists():
            # 对于单个文件，我们需要重新创建img2img pipeline
            base_pipeline = self.pipelines["text2img"]
            self.pipelines["img2img"] = StableDiffusionImg2ImgPipeline(
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                tokenizer=base_pipeline.tokenizer,
                unet=base_pipeline.unet,
                scheduler=base_pipeline.scheduler,
                safety_checker=None,
                feature_extractor=base_pipeline.feature_extractor
            )
        else:
            self.pipelines["img2img"] = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
        
        # 移动到设备并优化
        for pipeline in [self.pipelines["text2img"], self.pipelines["img2img"]]:
            pipeline = pipeline.to(self.device)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
            pipeline = device_manager.optimize_for_device(pipeline)
    
    def _load_sdxl_pipeline(self):
        """加载SDXL管道"""
        try:
            model_id = MODEL_CONFIGS["stable_diffusion"]["sdxl_model"]
            
            self.pipelines["text2img_xl"] = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )
            
            self.pipelines["img2img_xl"] = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )
            
            # 移动到设备并优化
            for pipeline in [self.pipelines["text2img_xl"], self.pipelines["img2img_xl"]]:
                pipeline = pipeline.to(self.device)
                pipeline = device_manager.optimize_for_device(pipeline)
                
        except Exception as e:
            logger.warning(f"Failed to load SDXL models: {e}")
    
    def _load_controlnet_models(self):
        """加载ControlNet模型"""
        try:
            controlnet_configs = MODEL_CONFIGS["stable_diffusion"]["controlnet_models"]
            
            for name, model_id in controlnet_configs.items():
                self.controlnet_models[name] = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype
                ).to(self.device)
                
        except Exception as e:
            logger.warning(f"Failed to load ControlNet models: {e}")
    
    def text2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        use_sdxl: bool = False
    ) -> List[Image.Image]:
        """文生图"""
        try:
            pipeline_name = "text2img_xl" if use_sdxl else "text2img"
            
            if pipeline_name not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_name} not available")
            
            pipeline = self.pipelines[pipeline_name]
            
            # 设置生成器
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # 生成图片
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator
                )
            
            images = result.images
            
            # 保存图片
            self._save_images(images, prompt, "text2img")
            
            return images
            
        except Exception as e:
            logger.error(f"Text2Img generation failed: {e}")
            raise
    
    def img2img(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        negative_prompt: str = "",
        strength: float = 0.7,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        use_sdxl: bool = False
    ) -> List[Image.Image]:
        """图生图"""
        try:
            pipeline_name = "img2img_xl" if use_sdxl else "img2img"
            
            if pipeline_name not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_name} not available")
            
            pipeline = self.pipelines[pipeline_name]
            
            # 处理输入图片
            if isinstance(image, str):
                input_image = Image.open(image).convert("RGB")
            else:
                input_image = image
            
            # 设置生成器
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # 生成图片
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator
                )
            
            images = result.images
            
            # 保存图片
            self._save_images(images, prompt, "img2img")
            
            return images
            
        except Exception as e:
            logger.error(f"Img2Img generation failed: {e}")
            raise
    
    def controlnet_generate(
        self,
        prompt: str,
        control_image: Union[str, Image.Image],
        control_type: str = "canny",
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """ControlNet生成"""
        try:
            if control_type not in self.controlnet_models:
                raise ValueError(f"ControlNet type {control_type} not available")
            
            controlnet = self.controlnet_models[control_type]
            
            # 创建ControlNet管道
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                MODEL_CONFIGS["stable_diffusion"]["default_model"],
                controlnet=controlnet,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # 处理控制图片
            if isinstance(control_image, str):
                control_img = Image.open(control_image).convert("RGB")
            else:
                control_img = control_image
            
            # 设置生成器
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # 生成图片
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_img,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator
                )
            
            images = result.images
            
            # 保存图片
            self._save_images(images, prompt, f"controlnet_{control_type}")
            
            return images
            
        except Exception as e:
            logger.error(f"ControlNet generation failed: {e}")
            raise
    
    def _save_images(self, images: List[Image.Image], prompt: str, prefix: str):
        """保存图片"""
        try:
            output_dir = OUTPUT_DIRS["images"]
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt[:50]
            
            for i, image in enumerate(images):
                filename = f"{prefix}_{safe_prompt}_{i+1}.png"
                filepath = output_dir / filename
                image.save(filepath)
                logger.info(f"Saved image: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save images: {e}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用模型列表"""
        return {
            "pipelines": list(self.pipelines.keys()),
            "controlnet_types": list(self.controlnet_models.keys())
        }
    
    def clear_cache(self):
        """清理缓存"""
        device_manager.clear_cache()
