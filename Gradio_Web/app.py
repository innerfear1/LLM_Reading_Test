"""
Gradio统一工作台 - 集成文生图、图生图、视频字幕、提示词助理
"""

import gradio as gr
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Any
import json

# 导入模块
from modules.stable_diffusion_module import StableDiffusionModule
from modules.blip_video_module import BLIPVideoModule
from modules.nano_gpt_module import NanoGPTModule
from configs.settings import OUTPUT_DIRS, DEFAULT_PARAMS, SECURITY_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模块实例
sd_module = None
blip_module = None
gpt_module = None


def initialize_modules():
    """初始化所有模块"""
    global sd_module, blip_module, gpt_module
    
    try:
        logger.info("Initializing modules...")
        
        # 初始化Stable Diffusion模块
        sd_module = StableDiffusionModule()
        logger.info("✅ Stable Diffusion module initialized")
        
        # 初始化BLIP Video模块
        blip_module = BLIPVideoModule()
        logger.info("✅ BLIP Video module initialized")
        
        # 初始化Nano GPT模块
        gpt_module = NanoGPTModule()
        logger.info("✅ Nano GPT module initialized")
        
        logger.info("🎉 All modules initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize modules: {e}")
        raise


def validate_prompt(prompt: str) -> bool:
    """验证提示词安全性"""
    if not prompt or len(prompt.strip()) == 0:
        return False
    
    if len(prompt) > SECURITY_CONFIG["max_prompt_length"]:
        return False
    
    prompt_lower = prompt.lower()
    for keyword in SECURITY_CONFIG["blocked_keywords"]:
        if keyword in prompt_lower:
            return False
    
    return True


def text2img_interface(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    seed: Optional[int],
    use_sdxl: bool
) -> List[str]:
    """文生图接口"""
    try:
        if not validate_prompt(prompt):
            return [None], "❌ 提示词不符合安全要求"
        
        if sd_module is None:
            return [None], "❌ Stable Diffusion模块未初始化"
        
        # 生成图片
        images = sd_module.text2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            seed=seed,
            use_sdxl=use_sdxl
        )
        
        # 返回图片路径
        image_paths = []
        for img in images:
            # 保存到临时目录
            temp_path = OUTPUT_DIRS["images"] / f"temp_{len(image_paths)}.png"
            img.save(temp_path)
            image_paths.append(str(temp_path))
        
        return image_paths, f"✅ 成功生成 {len(images)} 张图片"
        
    except Exception as e:
        logger.error(f"Text2Img failed: {e}")
        return [None], f"❌ 生成失败: {str(e)}"


def img2img_interface(
    prompt: str,
    negative_prompt: str,
    image: Any,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    seed: Optional[int],
    use_sdxl: bool
) -> List[str]:
    """图生图接口"""
    try:
        if not validate_prompt(prompt):
            return [None], "❌ 提示词不符合安全要求"
        
        if image is None:
            return [None], "❌ 请上传输入图片"
        
        if sd_module is None:
            return [None], "❌ Stable Diffusion模块未初始化"
        
        # 生成图片
        images = sd_module.img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            seed=seed,
            use_sdxl=use_sdxl
        )
        
        # 返回图片路径
        image_paths = []
        for img in images:
            temp_path = OUTPUT_DIRS["images"] / f"temp_{len(image_paths)}.png"
            img.save(temp_path)
            image_paths.append(str(temp_path))
        
        return image_paths, f"✅ 成功生成 {len(images)} 张图片"
        
    except Exception as e:
        logger.error(f"Img2Img failed: {e}")
        return [None], f"❌ 生成失败: {str(e)}"


def video_caption_interface(
    video: Any,
    fps: float,
    max_frames: int,
    max_length: int,
    export_srt: bool,
    export_vtt: bool
) -> Tuple[str, str, str]:
    """视频字幕接口"""
    try:
        if video is None:
            return None, None, "❌ 请上传视频文件"
        
        if blip_module is None:
            return None, None, "❌ BLIP Video模块未初始化"
        
        # 处理视频
        results = blip_module.process_video_with_captions(
            video_path=video,
            fps=fps,
            max_frames=max_frames,
            export_formats=(["srt"] if export_srt else []) + (["vtt"] if export_vtt else [])
        )
        
        # 生成字幕文本
        captions_text = ""
        for caption_data in results["captions"]:
            captions_text += f"[{caption_data['timestamp']:.1f}s] {caption_data['caption']}\n"
        
        # 返回文件路径
        srt_file = None
        vtt_file = None
        
        for file_path in results["exported_files"]:
            if file_path.endswith('.srt'):
                srt_file = file_path
            elif file_path.endswith('.vtt'):
                vtt_file = file_path
        
        return srt_file, vtt_file, f"✅ 成功生成 {len(results['captions'])} 个字幕"
        
    except Exception as e:
        logger.error(f"Video caption failed: {e}")
        return None, None, f"❌ 处理失败: {str(e)}"


def prompt_assistant_interface(
    base_prompt: str,
    operation: str,
    style: str,
    max_length: int,
    temperature: float
) -> Tuple[str, str, str]:
    """提示词助理接口"""
    try:
        if not base_prompt or len(base_prompt.strip()) == 0:
            return "", "", "❌ 请输入基础提示词"
        
        if gpt_module is None:
            return "", "", "❌ Nano GPT模块未初始化"
        
        # 执行操作
        if operation == "expand":
            results = gpt_module.expand_prompt(
                base_prompt, 
                style=style, 
                max_length=max_length, 
                temperature=temperature
            )
        elif operation == "optimize":
            results = gpt_module.optimize_prompt(
                base_prompt, 
                target_style=style, 
                max_length=max_length
            )
        elif operation == "negative":
            results = gpt_module.generate_negative_prompt(base_prompt)
        else:
            results = [base_prompt]
        
        # 分析提示词质量
        analysis = gpt_module.analyze_prompt_quality(base_prompt)
        
        # 格式化输出
        expanded_prompts = "\n\n".join([f"版本 {i+1}:\n{result}" for i, result in enumerate(results)])
        
        analysis_text = f"""
提示词分析:
- 长度: {analysis['length']} 字符
- 词数: {analysis['word_count']} 个词
- 包含风格: {'✅' if analysis['has_style'] else '❌'}
- 包含质量: {'✅' if analysis['has_quality'] else '❌'}
- 包含光照: {'✅' if analysis['has_lighting'] else '❌'}
- 包含构图: {'✅' if analysis['has_composition'] else '❌'}

建议:
{chr(10).join(f"- {suggestion}" for suggestion in analysis['suggestions'])}
"""
        
        return expanded_prompts, analysis_text, f"✅ 成功处理提示词"
        
    except Exception as e:
        logger.error(f"Prompt assistant failed: {e}")
        return "", "", f"❌ 处理失败: {str(e)}"


def create_interface():
    """创建Gradio界面"""
    
    # 自定义CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    """
    
    with gr.Blocks(css=css, title="AI创作工作台") as app:
        gr.Markdown("""
        # 🎨 AI创作工作台
        **集成文生图、图生图、视频字幕、提示词助理的一体化平台**
        """)
        
        with gr.Tabs():
            # 文生图标签页
            with gr.Tab("🎨 文生图"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text2img_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="描述你想要生成的图片...",
                            lines=3
                        )
                        text2img_negative = gr.Textbox(
                            label="负面提示词",
                            placeholder="描述不想要的内容...",
                            lines=2
                        )
                        
                        with gr.Row():
                            text2img_width = gr.Slider(256, 1024, 512, step=64, label="宽度")
                            text2img_height = gr.Slider(256, 1024, 512, step=64, label="高度")
                        
                        with gr.Row():
                            text2img_steps = gr.Slider(10, 50, 20, step=5, label="推理步数")
                            text2img_guidance = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="引导强度")
                        
                        with gr.Row():
                            text2img_num = gr.Slider(1, 4, 1, step=1, label="生成数量")
                            text2img_seed = gr.Number(label="种子", precision=0)
                        
                        text2img_sdxl = gr.Checkbox(label="使用SDXL模型", value=False)
                        
                        text2img_btn = gr.Button("🎨 生成图片", variant="primary")
                    
                    with gr.Column(scale=1):
                        text2img_output = gr.Gallery(label="生成的图片")
                        text2img_status = gr.Textbox(label="状态", interactive=False)
            
            # 图生图标签页
            with gr.Tab("🔄 图生图"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img2img_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="描述你想要的变换效果...",
                            lines=3
                        )
                        img2img_negative = gr.Textbox(
                            label="负面提示词",
                            placeholder="描述不想要的内容...",
                            lines=2
                        )
                        img2img_input = gr.Image(label="输入图片", type="pil")
                        
                        with gr.Row():
                            img2img_strength = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="变换强度")
                            img2img_steps = gr.Slider(10, 50, 20, step=5, label="推理步数")
                        
                        with gr.Row():
                            img2img_guidance = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="引导强度")
                            img2img_num = gr.Slider(1, 4, 1, step=1, label="生成数量")
                        
                        with gr.Row():
                            img2img_seed = gr.Number(label="种子", precision=0)
                            img2img_sdxl = gr.Checkbox(label="使用SDXL模型", value=False)
                        
                        img2img_btn = gr.Button("🔄 生成图片", variant="primary")
                    
                    with gr.Column(scale=1):
                        img2img_output = gr.Gallery(label="生成的图片")
                        img2img_status = gr.Textbox(label="状态", interactive=False)
            
            # 视频字幕标签页
            with gr.Tab("🎬 视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="输入视频")
                        
                        with gr.Row():
                            video_fps = gr.Slider(0.1, 2.0, 1.0, step=0.1, label="采样帧率")
                            video_max_frames = gr.Slider(10, 200, 50, step=10, label="最大帧数")
                        
                        video_max_length = gr.Slider(20, 200, 100, step=10, label="字幕最大长度")
                        
                        with gr.Row():
                            video_export_srt = gr.Checkbox(label="导出SRT", value=True)
                            video_export_vtt = gr.Checkbox(label="导出VTT", value=True)
                        
                        video_btn = gr.Button("🎬 生成字幕", variant="primary")
                    
                    with gr.Column(scale=1):
                        video_srt_output = gr.File(label="SRT字幕文件")
                        video_vtt_output = gr.File(label="VTT字幕文件")
                        video_status = gr.Textbox(label="状态", interactive=False)
            
            # 提示词助理标签页
            with gr.Tab("🤖 提示词助理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_base = gr.Textbox(
                            label="基础提示词",
                            placeholder="输入你的基础提示词...",
                            lines=3
                        )
                        
                        with gr.Row():
                            prompt_operation = gr.Radio(
                                choices=["expand", "optimize", "negative"],
                                value="expand",
                                label="操作类型"
                            )
                            prompt_style = gr.Dropdown(
                                choices=["detailed", "artistic", "photorealistic", "fantasy", "sci-fi", "anime", "vintage"],
                                value="detailed",
                                label="风格"
                            )
                        
                        with gr.Row():
                            prompt_max_length = gr.Slider(50, 300, 150, step=10, label="最大长度")
                            prompt_temperature = gr.Slider(0.1, 1.5, 0.8, step=0.1, label="创造性")
                        
                        prompt_btn = gr.Button("🤖 处理提示词", variant="primary")
                    
                    with gr.Column(scale=1):
                        prompt_output = gr.Textbox(
                            label="处理结果",
                            lines=10,
                            interactive=False
                        )
                        prompt_analysis = gr.Textbox(
                            label="提示词分析",
                            lines=8,
                            interactive=False
                        )
                        prompt_status = gr.Textbox(label="状态", interactive=False)
        
        # 绑定事件
        text2img_btn.click(
            fn=text2img_interface,
            inputs=[
                text2img_prompt, text2img_negative, text2img_width, text2img_height,
                text2img_steps, text2img_guidance, text2img_num, text2img_seed, text2img_sdxl
            ],
            outputs=[text2img_output, text2img_status]
        )
        
        img2img_btn.click(
            fn=img2img_interface,
            inputs=[
                img2img_prompt, img2img_negative, img2img_input, img2img_strength,
                img2img_steps, img2img_guidance, img2img_num, img2img_seed, img2img_sdxl
            ],
            outputs=[img2img_output, img2img_status]
        )
        
        video_btn.click(
            fn=video_caption_interface,
            inputs=[
                video_input, video_fps, video_max_frames, video_max_length,
                video_export_srt, video_export_vtt
            ],
            outputs=[video_srt_output, video_vtt_output, video_status]
        )
        
        prompt_btn.click(
            fn=prompt_assistant_interface,
            inputs=[
                prompt_base, prompt_operation, prompt_style, prompt_max_length, prompt_temperature
            ],
            outputs=[prompt_output, prompt_analysis, prompt_status]
        )
    
    return app


def main():
    """主函数"""
    try:
        # 初始化模块
        initialize_modules()
        
        # 创建界面
        app = create_interface()
        
        # 启动应用
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
