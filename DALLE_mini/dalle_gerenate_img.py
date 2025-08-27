# dalle_mini by inner - using min-dalle approach

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from min_dalle import MinDalle
import torch


class Text2imgProcess:
    def __init__(self, model_path="/Users/innerwq/Desktop/dalle_model"):
        """
        初始化DALLE-mini模型
        Args:
            model_path: 本地模型路径
        """
        self.model_path = model_path
        self.model = None
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Load models in initialization
        self.load_models(model_path)
    
    def load_models(self, model_path):
        """加载DALLE-mini"""
        try:
            print("Loading MinDalle model...")
            
            # Initialize MinDalle model
            self.model = MinDalle(
                models_root=model_path,
                dtype=torch.float32,
                device=self.device,
                is_mega=False,  # Use smaller model for faster generation
                is_reusable=True  # Keep model in memory for reuse
            )
            
            print("Models loaded successfully!")
            print(f"Using device: {self.device.upper()}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please ensure you have the required dependencies installed:")
            print("pip install min-dalle torch")
    
    def generate_image(self, prompt, n_predictions=2, condition_scale=15.0):
        """
        根据文本提示生成图像
        Args:
            prompt: 文本提示
            n_predictions: 生成图像数量
            condition_scale: 条件缩放参数，控制生成质量
        Returns:
            生成的图像列表
        """
        if self.model is None:
            print("Models not loaded. Please call load_models() first.")
            return []
        
        try:
            print(f"Generating {n_predictions} images for prompt: '{prompt}'")
            
            # Generate images using MinDalle
            images = []
            for i in range(n_predictions):
                print(f"Generating image {i+1}/{n_predictions}...")
                
                # Generate image with different seed for each iteration
                seed = torch.randint(0, 2**32, (1,)).item()
                
                image = self.model.generate_image(
                    text=prompt,
                    seed=seed,
                    grid_size=1,  # Generate single image
                    is_seamless=False,
                    temperature=0.7,
                    top_k=128,
                    supercondition_factor=condition_scale
                )
                
                # resize to 512x512 for consistent output size
                try:
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                except Exception:
                    image = image.resize((512, 512))
                
                images.append(image)
            
            print("Image generation completed!")
            return images
            
        except Exception as e:
            print(f"Error generating images: {e}")
            return []
    
    def process_prompt(self, prompt, n_predictions=4):
        """
        处理文本提示并生成图像
        Args:
            prompt: 文本提示
            n_predictions: 生成图像数量
        """
        if not prompt or prompt.strip() == "":
            print("Please provide a valid prompt.")
            return []
        
        images = self.generate_image(prompt, n_predictions)
        
        if images:
            print(f"Successfully generated {len(images)} images!")
            self.display_images(images, prompt)
            self.save_images(images, prompt)
        
        return images
    
    def save_images(self, images, prompt, output_dir="/Users/innerwq/Desktop/test_demo/dalle-generate"):
        """
        保存生成的图像
        Args:
            images: 图像列表
            prompt: 原始提示
            output_dir: 输出目录
        """
        if not images:
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean prompt for filename
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt[:50]  # Limit length
        
        print(f"Saving images to {output_dir}/")
        
        for i, image in enumerate(images):
            filename = f"{safe_prompt}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")
      
    def display_images(self, images, prompt):
        """
        显示生成的图像
        Args:
            images: 图像列表
            prompt: 原始提示
        """
        if not images:
            return
        
        n_images = len(images)
        cols = min(n_images, 4)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # 统一将 axes 转为一维数组，便于索引
        axes_array = np.array(axes).reshape(-1) if isinstance(axes, (list, np.ndarray)) else np.array([axes])
        
        for i, image in enumerate(images):
            ax = axes_array[i]
            ax.imshow(image)
            ax.set_title(f"Image {i+1}")
            ax.axis('off')
        
        # 隐藏多余子图
        for j in range(len(images), len(axes_array)):
            axes_array[j].axis('off')
        
        plt.suptitle(f"Generated images for: '{prompt}'", fontsize=16)
        plt.tight_layout()
        plt.show()
        

def main():
    """
    主函数,演示DALLE-mini的使用
    """
    # Initialize the processor
    processor = Text2imgProcess()
    
    # Example prompts
    # TODO @https://github.com/o390-Pxk  添加中文生图的方法
    prompts = [
        "A futuristic city at night filled with neon lights, flying cars, and giant holographic billboards. Cyberpunk style, low-angle view, with a sci-fi and metallic atmosphere.",
        "An impressionist oil painting of a girl in a white dress running through a golden wheat field. The sky is bright blue, with loose, energetic brushstrokes. Vibrant colors and strong light contrast convey a lively atmosphere.",
        "A young woman in a vintage white dress standing under cherry blossoms, smiling at the camera. The background is filled with pale pink flowers. Soft morning light. Photorealistic style, high resolution."
    ]
    
    print("MinDalle Image Generation Demo")
    print("=" * 40)
    
    for prompt in prompts:
        print(f"\nProcessing prompt: '{prompt}'")
        images = processor.process_prompt(prompt, n_predictions=1)
        
        if images:
            print(f"Generated {len(images)} images successfully!")
        else:
            print("Failed to generate images.")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
