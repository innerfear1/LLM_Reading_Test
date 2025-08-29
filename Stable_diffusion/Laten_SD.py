# Stable Diffusion for Mac M-series
# Based on Stability-AI official scripts
# By inner

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class ModelSetting:
    """模型设置类"""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = self._get_device()
        self.pipeline = None
        self.setup_pipeline()
    
    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def setup_pipeline(self):
        try:
            print(f"Loading model {self.model_id} on device {self.device}")
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            if self.device == "mps":
                self.pipeline.enable_attention_slicing()
            
            print("Pipeline loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            raise


class Text2Img:
    """文生图类"""
    
    def __init__(self, model_setting: ModelSetting):
        self.model_setting = model_setting
        self.pipeline = model_setting.pipeline
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        output_dir: str = "./outputs"
    ) -> List[Image.Image]:
        
        if seed is not None:
            torch.manual_seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"Generating images for prompt: {prompt}")
            
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                return_dict=False
            )[0]
            
            print(f"Generated {len(images)} images successfully!")
            self._save_images(images, prompt, output_dir)
            return images
            
        except Exception as e:
            print(f"Generation failed: {e}")
            raise
    
    def _save_images(self, images: List[Image.Image], prompt: str, output_dir: str):
        for i, image in enumerate(images):
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt[:50]
            
            filename = f"{safe_prompt}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")


class Img2Img:
    """图生图类"""
    
    def __init__(self, model_setting: ModelSetting):
        self.model_setting = model_setting
        self.pipeline = model_setting.pipeline
    
    def generate(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        negative_prompt: str = "",
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        output_dir: str = "./outputs"
    ) -> List[Image.Image]:
        
        if seed is not None:
            torch.manual_seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(image, str):
            input_image = Image.open(image).convert("RGB")
        else:
            input_image = image
        
        try:
            print(f"Generating images for prompt: {prompt}")
            
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                return_dict=False
            )[0]
            
            print(f"Generated {len(images)} images successfully!")
            self._save_images(images, prompt, output_dir)
            return images
            
        except Exception as e:
            print(f"Generation failed: {e}")
            raise
    
    def _save_images(self, images: List[Image.Image], prompt: str, output_dir: str):
        for i, image in enumerate(images):
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt[:50]
            
            filename = f"img2img_{safe_prompt}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")


class StableDiffusionApp:
    """Stable Diffusion应用主类"""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_setting = ModelSetting(model_id)
        self.text2img = Text2Img(self.model_setting)
        self.img2img = Img2Img(self.model_setting)
    
    def generate_text2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        return self.text2img.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            seed=seed
        )
    
    def generate_img2img(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        negative_prompt: str = "",
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        return self.img2img.generate(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            seed=seed
        )
    
    def display_images(self, images: List[Image.Image], title: str = "Generated Images"):
        if not images:
            return
        
        n_images = len(images)
        cols = min(n_images, 3)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_array = np.array(axes).reshape(-1)
        
        for i, image in enumerate(images):
            ax = axes_array[i]
            ax.imshow(image)
            ax.set_title(f"Image {i+1}")
            ax.axis('off')
        
        for j in range(len(images), len(axes_array)):
            axes_array[j].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()


def main():
    """主函数演示"""
    print("🚀 Stable Diffusion for Mac M-series")
    print("=" * 50)
    
    app = StableDiffusionApp()
    
    prompts = [
        "A beautiful sunset over mountains, photorealistic, high resolution",
        "A cute cat sitting in a garden with flowers, detailed, sharp focus",
        "A futuristic city with flying cars and neon lights, cyberpunk style"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎨 Generating image {i+1}/3...")
        print(f"Prompt: {prompt}")
        
        try:
            images = app.generate_text2img(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted",
                num_inference_steps=20,
                guidance_scale=7.5,
                num_images_per_prompt=1,
                seed=42 + i
            )
            
            app.display_images(images, f"Text2Img - {prompt[:30]}...")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print("\n✅ Demo completed!")


if __name__ == '__main__':
    main()