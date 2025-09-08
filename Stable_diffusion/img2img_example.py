#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图生图使用示例
Image-to-Image Generation Example

使用方法:
1. 直接运行: python img2img_example.py
2. 或者导入使用: from img2img_example import simple_img2img
"""

import os
from PIL import Image
from Laten_SD import StableDiffusionApp


def simple_img2img_example():
    """简单的图生图示例"""
    print("🎨 简单图生图示例")
    print("=" * 40)
    
    # 初始化应用
    app = StableDiffusionApp()
    
    # 示例1: 使用现有图片
    print("\n📸 示例1: 使用现有图片")
    
    # 如果你有现有图片，可以这样使用:
    # image_path = "path/to/your/image.jpg"
    # if os.path.exists(image_path):
    #     result = app.generate_img2img(
    #         prompt="transform into anime style",
    #         image=image_path,
    #         strength=0.7,
    #         num_inference_steps=30
    #     )
    
    # 示例2: 先生成基础图片，再进行图生图
    print("\n🎭 示例2: 生成基础图片后进行图生图")
    
    # 步骤1: 生成基础图片
    print("Step 1: 生成基础图片...")
    base_images = app.generate_text2img(
        prompt="a simple portrait of a person, neutral expression, clean background",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=20,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        seed=123
    )
    
    if not base_images:
        print("❌ 基础图片生成失败")
        return
    
    base_image = base_images[0]
    print("✅ 基础图片生成成功!")
    
    # 步骤2: 进行图生图变换
    print("\nStep 2: 进行图生图变换...")
    
    # 变换1: 赛博朋克风格
    print("🎭 变换1: 赛博朋克风格")
    cyberpunk_result = app.generate_img2img(
        prompt="transform into cyberpunk style with neon lights and futuristic elements",
        image=base_image,
        negative_prompt="blurry, low quality, distorted, ugly",
        strength=0.7,  # 变换强度 (0.0-1.0)
        guidance_scale=7.5,
        num_inference_steps=30,
        num_images_per_prompt=1,
        seed=456
    )
    
    if cyberpunk_result:
        print("✅ 赛博朋克变换完成!")
        app.display_images(cyberpunk_result, "Cyberpunk Style")
    
    # 变换2: 古典油画风格
    print("\n🎨 变换2: 古典油画风格")
    painting_result = app.generate_img2img(
        prompt="make it look like a classical oil painting with rich colors and brushstrokes",
        image=base_image,
        negative_prompt="blurry, low quality, distorted, ugly",
        strength=0.8,  # 更高的变换强度
        guidance_scale=7.5,
        num_inference_steps=30,
        num_images_per_prompt=1,
        seed=789
    )
    
    if painting_result:
        print("✅ 古典油画变换完成!")
        app.display_images(painting_result, "Classical Painting Style")


def custom_img2img(prompt: str, image_path: str, strength: float = 0.7):
    """自定义图生图函数"""
    print(f"🎨 自定义图生图: {prompt}")
    print("=" * 40)
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return None
    
    # 初始化应用
    app = StableDiffusionApp()
    
    try:
        # 进行图生图
        result = app.generate_img2img(
            prompt=prompt,
            image=image_path,
            negative_prompt="blurry, low quality, distorted, ugly",
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=30,
            num_images_per_prompt=1,
            seed=42
        )
        
        if result:
            print("✅ 图生图完成!")
            app.display_images(result, f"Custom: {prompt[:30]}...")
            return result
        else:
            print("❌ 图生图失败")
            return None
            
    except Exception as e:
        print(f"❌ 图生图出错: {e}")
        return None


def batch_img2img(image_path: str, prompts: list, strength: float = 0.7):
    """批量图生图"""
    print(f"🎨 批量图生图: {len(prompts)} 个变换")
    print("=" * 40)
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    app = StableDiffusionApp()
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎭 变换 {i+1}/{len(prompts)}: {prompt}")
        
        try:
            result = app.generate_img2img(
                prompt=prompt,
                image=image_path,
                negative_prompt="blurry, low quality, distorted, ugly",
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=30,
                num_images_per_prompt=1,
                seed=100 + i
            )
            
            if result:
                print(f"✅ 变换 {i+1} 完成!")
                app.display_images(result, f"Transform {i+1}: {prompt[:20]}...")
                results.append(result)
            else:
                print(f"❌ 变换 {i+1} 失败")
                
        except Exception as e:
            print(f"❌ 变换 {i+1} 出错: {e}")
    
    print(f"\n✅ 批量变换完成! 成功: {len(results)}/{len(prompts)}")
    return results


def main():
    """主函数"""
    print("🚀 图生图使用示例")
    print("=" * 50)
    
    print("\n请选择要运行的示例:")
    print("1. 简单图生图示例")
    print("2. 自定义图生图")
    print("3. 批量图生图")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            simple_img2img_example()
        
        elif choice == "2":
            print("\n📸 自定义图生图")
            image_path = input("请输入图片路径: ").strip()
            prompt = input("请输入变换提示词: ").strip()
            strength = float(input("请输入变换强度 (0.0-1.0, 默认0.7): ").strip() or "0.7")
            
            if image_path and prompt:
                custom_img2img(prompt, image_path, strength)
            else:
                print("❌ 请提供图片路径和提示词")
        
        elif choice == "3":
            print("\n🎨 批量图生图")
            image_path = input("请输入图片路径: ").strip()
            
            if image_path:
                # 预设的变换提示词
                prompts = [
                    "transform into anime style, colorful and vibrant",
                    "make it look like a watercolor painting with soft colors",
                    "add dramatic lighting and cinematic atmosphere",
                    "transform into cyberpunk style with neon lights",
                    "make it look like a vintage photograph with sepia tones"
                ]
                
                print(f"将使用以下 {len(prompts)} 个变换:")
                for i, prompt in enumerate(prompts, 1):
                    print(f"  {i}. {prompt}")
                
                confirm = input("\n确认开始批量变换? (y/n): ").strip().lower()
                if confirm == 'y':
                    batch_img2img(image_path, prompts, 0.7)
                else:
                    print("❌ 取消批量变换")
            else:
                print("❌ 请提供图片路径")
        
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
    
    print("\n✅ 程序结束!")


if __name__ == '__main__':
    main()
