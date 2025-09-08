# Stable Diffusion for Mac M-series
# Based on Stability-AI official scripts
# By inner

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union
import matplotlib.pyplot as plt

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import...")
    from diffusers import StableDiffusionPipeline
    from diffusers.schedulers import DPMSolverMultistepScheduler
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
            
            # 对于MPS设备，使用float32避免精度问题
            if self.device == "mps":
                torch_dtype = torch.float32
                print("Using float32 for MPS device to avoid precision issues")
            elif self.device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # 设置调度器
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # MPS优化
            if self.device == "mps":
                self.pipeline.enable_attention_slicing()
                # 禁用内存高效注意力，避免MPS问题
                try:
                    self.pipeline.disable_memory_efficient_attention()
                except:
                    pass
            
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
            print(f"Device: {self.model_setting.device}")
            print(f"Parameters: steps={num_inference_steps}, guidance={guidance_scale}, size={height}x{width}")
            
            # 确保输入参数正确
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.model_setting.device).manual_seed(seed)
            
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    return_dict=False
                )
            
            images = result[0]
            
            # 检查生成的图像
            if images and len(images) > 0:
                first_image = images[0]
                print(f"Generated image shape: {first_image.size}")
                print(f"Generated image mode: {first_image.mode}")
                
                # 检查图像是否全黑
                img_array = np.array(first_image)
                if np.all(img_array == 0):
                    print("⚠️ Warning: Generated image appears to be all black!")
                    print("This might be due to model loading or device compatibility issues.")
                else:
                    print(f"✅ Image generation successful! Mean pixel value: {np.mean(img_array):.2f}")
            
            print(f"Generated {len(images)} images successfully!")
            self._save_images(images, prompt, output_dir)
            return images
            
        except Exception as e:
            print(f"Generation failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
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


def test_simple_generation():
    """简单测试函数"""
    print("🧪 Testing simple image generation...")
    
    try:
        app = StableDiffusionApp()
        
        # 使用简单的提示词测试
        prompt = "a red apple on a white table"
        
        print(f"Testing with prompt: {prompt}")
        
        images = app.generate_text2img(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            num_inference_steps=10,  # 减少步数加快测试
            guidance_scale=7.5,
            num_images_per_prompt=1,
            seed=42
        )
        
        if images:
            print("✅ Test generation successful!")
            app.display_images(images, "Test Image")
            return True
        else:
            print("❌ No images generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def demo_img2img():
    """图生图演示函数"""
    print("🎨 Image-to-Image Generation Demo")
    print("=" * 50)
    
    try:
        app = StableDiffusionApp()
        
        # 首先生成一张基础图片作为输入
        print("📸 Step 1: Generating base image...")
        base_images = app.generate_text2img(
            prompt="a simple portrait of a person, neutral expression, clean background",
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=20,
            guidance_scale=7.5,
            num_images_per_prompt=1,
            seed=123
        )
        
        if not base_images:
            print("❌ Failed to generate base image")
            return
        
        base_image = base_images[0]
        print("✅ Base image generated successfully!")
        
        # 保存基础图片
        base_image.save("./outputs/base_image.png")
        print("💾 Base image saved as './outputs/base_image.png'")
        
        # 图生图示例
        img2img_examples = [
            {
                "prompt": "transform into a cyberpunk style character with neon lights and futuristic elements",
                "strength": 0.7,
                "description": "Cyberpunk transformation"
            },
            {
                "prompt": "make it look like a vintage oil painting with classical art style",
                "strength": 0.8,
                "description": "Classical painting style"
            },
            {
                "prompt": "add fantasy elements, magical aura, and mystical background",
                "strength": 0.6,
                "description": "Fantasy transformation"
            }
        ]
        
        for i, example in enumerate(img2img_examples):
            print(f"\n🎭 Step {i+2}: {example['description']}")
            print(f"Prompt: {example['prompt']}")
            print(f"Strength: {example['strength']}")
            
            try:
                result_images = app.generate_img2img(
                    prompt=example['prompt'],
                    image=base_image,  # 使用生成的基础图片
                    negative_prompt="blurry, low quality, distorted, ugly",
                    strength=example['strength'],
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    num_images_per_prompt=1,
                    seed=456 + i
                )
                
                if result_images:
                    print(f"✅ {example['description']} completed!")
                    app.display_images(result_images, f"Img2Img - {example['description']}")
                else:
                    print(f"❌ {example['description']} failed")
                    
            except Exception as e:
                print(f"❌ {example['description']} failed: {e}")
        
        print("\n✅ Image-to-Image demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_img2img_with_existing_image(image_path: str):
    """使用现有图片进行图生图演示"""
    print(f"🎨 Image-to-Image with existing image: {image_path}")
    print("=" * 50)
    
    try:
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return
        
        app = StableDiffusionApp()
        
        # 加载现有图片
        input_image = Image.open(image_path).convert("RGB")
        print(f"📸 Loaded image: {input_image.size}")
        
        # 图生图变换示例
        transformations = [
            {
                "prompt": "transform into anime style, colorful and vibrant",
                "strength": 0.7,
                "description": "Anime style"
            },
            {
                "prompt": "make it look like a watercolor painting with soft colors",
                "strength": 0.8,
                "description": "Watercolor style"
            },
            {
                "prompt": "add dramatic lighting and cinematic atmosphere",
                "strength": 0.5,
                "description": "Cinematic style"
            }
        ]
        
        for i, transform in enumerate(transformations):
            print(f"\n🎭 Transformation {i+1}: {transform['description']}")
            print(f"Prompt: {transform['prompt']}")
            
            try:
                result_images = app.generate_img2img(
                    prompt=transform['prompt'],
                    image=input_image,
                    negative_prompt="blurry, low quality, distorted",
                    strength=transform['strength'],
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    num_images_per_prompt=1,
                    seed=789 + i
                )
                
                if result_images:
                    print(f"✅ {transform['description']} completed!")
                    app.display_images(result_images, f"Transform - {transform['description']}")
                else:
                    print(f"❌ {transform['description']} failed")
                    
            except Exception as e:
                print(f"❌ {transform['description']} failed: {e}")
        
        print("\n✅ Image transformation demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数演示"""
    print("🚀 Stable Diffusion for Mac M-series")
    print("=" * 50)
    
    # 显示选项菜单
    print("\n请选择要运行的功能:")
    print("1. 文生图 (Text-to-Image)")
    print("2. 图生图演示 (Image-to-Image Demo)")
    print("3. 使用现有图片进行图生图")
    print("4. 运行完整演示")
    
    try:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            print("\n🎨 文生图演示")
            print("=" * 30)
            app = StableDiffusionApp()
            
            prompts = [
                "An impressionist oil painting of a girl in a white dress running through a golden wheat field. The sky is bright blue, with loose, energetic brushstrokes. Vibrant colors and strong light contrast convey a lively atmosphere.",
                "A young woman in a vintage white dress standing under cherry blossoms, smiling at the camera. The background is filled with pale pink flowers. Soft morning light. Photorealistic style, high resolution."
            ]
            
            for i, prompt in enumerate(prompts):
                print(f"\n🎨 Generating image {i+1}/{len(prompts)}...")
                print(f"Prompt: {prompt}")
                
                try:
                    images = app.generate_text2img(
                        prompt=prompt,
                        negative_prompt="blurry, low quality, distorted",
                        num_inference_steps=40,
                        guidance_scale=7.5,
                        num_images_per_prompt=1,
                        seed=42 + i
                    )
                    
                    app.display_images(images, f"Text2Img - {prompt[:30]}...")
                    
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
        
        elif choice == "2":
            print("\n🎭 图生图演示")
            demo_img2img()
        
        elif choice == "3":
            print("\n📸 使用现有图片进行图生图")
            image_path = input("请输入图片路径: ").strip()
            if image_path:
                demo_img2img_with_existing_image(image_path)
            else:
                print("❌ 未提供图片路径")
        
        elif choice == "4":
            print("\n🎨 运行完整演示")
            print("=" * 30)
            
            # 文生图部分
            app = StableDiffusionApp()
            prompts = [
                "An impressionist oil painting of a girl in a white dress running through a golden wheat field. The sky is bright blue, with loose, energetic brushstrokes. Vibrant colors and strong light contrast convey a lively atmosphere.",
                "A young woman in a vintage white dress standing under cherry blossoms, smiling at the camera. The background is filled with pale pink flowers. Soft morning light. Photorealistic style, high resolution."
            ]
            
            for i, prompt in enumerate(prompts):
                print(f"\n🎨 Generating image {i+1}/{len(prompts)}...")
                print(f"Prompt: {prompt}")
                
                try:
                    images = app.generate_text2img(
                        prompt=prompt,
                        negative_prompt="blurry, low quality, distorted",
                        num_inference_steps=40,
                        guidance_scale=7.5,
                        num_images_per_prompt=1,
                        seed=42 + i
                    )
                    
                    app.display_images(images, f"Text2Img - {prompt[:30]}...")
                    
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
            
            # 图生图部分
            print("\n" + "=" * 50)
            demo_img2img()
        
        else:
            print("❌ 无效选择，请运行程序重新选择")
    
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
    
    print("\n✅ 程序结束!")


if __name__ == '__main__':
    main()