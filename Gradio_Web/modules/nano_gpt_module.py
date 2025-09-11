"""
Nano GPT模块 - 提示词生成、优化、扩写
"""

import torch
import logging
from typing import List, Optional, Dict, Any
import re

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from utils.device_manager import device_manager
from configs.settings import MODEL_CONFIGS

logger = logging.getLogger(__name__)


class NanoGPTModule:
    """Nano GPT模块"""
    
    def __init__(self):
        self.device = device_manager.get_device()
        self.torch_dtype = device_manager.get_torch_dtype()
        self.model = None
        self.tokenizer = None
        self._load_models()
    
    def _load_models(self):
        """加载GPT模型"""
        try:
            logger.info("Loading Nano GPT models...")
            
            # 优先使用本地模型
            local_model_path = MODEL_CONFIGS["nano_gpt"]["local_models"]["gpt2"]
            if local_model_path.exists():
                model_id = str(local_model_path)
                logger.info(f"Using local GPT model: {model_id}")
            else:
                model_id = MODEL_CONFIGS["nano_gpt"]["default_model"]
                logger.info(f"Using remote GPT model: {model_id}")
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
            self.model = GPT2LMHeadModel.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Nano GPT models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load GPT models: {e}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """生成文本"""
        try:
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # 移除原始prompt，只保留生成的部分
                generated_part = text[len(prompt):].strip()
                generated_texts.append(generated_part)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def expand_prompt(
        self,
        base_prompt: str,
        style: str = "detailed",
        max_length: int = 150,
        temperature: float = 0.8
    ) -> List[str]:
        """扩写提示词"""
        try:
            # 根据风格调整前缀
            style_prefixes = {
                "detailed": "A detailed description: ",
                "artistic": "An artistic prompt: ",
                "photorealistic": "A photorealistic prompt: ",
                "fantasy": "A fantasy art prompt: ",
                "sci-fi": "A sci-fi art prompt: ",
                "anime": "An anime style prompt: ",
                "vintage": "A vintage style prompt: "
            }
            
            prefix = style_prefixes.get(style, "A detailed prompt: ")
            full_prompt = prefix + base_prompt
            
            # 生成扩写
            expanded_texts = self.generate_text(
                full_prompt,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=3
            )
            
            # 清理和格式化输出
            cleaned_texts = []
            for text in expanded_texts:
                # 移除前缀
                cleaned = text.replace(prefix, "").strip()
                # 移除重复的base_prompt
                if cleaned.startswith(base_prompt):
                    cleaned = cleaned[len(base_prompt):].strip()
                # 确保以base_prompt开头
                if not cleaned.startswith(base_prompt):
                    cleaned = base_prompt + " " + cleaned
                cleaned_texts.append(cleaned)
            
            return cleaned_texts
            
        except Exception as e:
            logger.error(f"Prompt expansion failed: {e}")
            raise
    
    def optimize_prompt(
        self,
        prompt: str,
        target_style: str = "high_quality",
        max_length: int = 120
    ) -> List[str]:
        """优化提示词"""
        try:
            # 优化模板
            optimization_templates = {
                "high_quality": "Optimize this prompt for high quality: ",
                "artistic": "Make this prompt more artistic: ",
                "detailed": "Add more details to this prompt: ",
                "concise": "Make this prompt more concise: ",
                "professional": "Make this prompt more professional: "
            }
            
            template = optimization_templates.get(target_style, "Optimize this prompt: ")
            full_prompt = template + prompt
            
            # 生成优化版本
            optimized_texts = self.generate_text(
                full_prompt,
                max_length=max_length,
                temperature=0.6,
                num_return_sequences=2
            )
            
            # 清理输出
            cleaned_texts = []
            for text in optimized_texts:
                cleaned = text.replace(template, "").strip()
                if not cleaned:
                    cleaned = prompt  # 如果生成失败，返回原提示词
                cleaned_texts.append(cleaned)
            
            return cleaned_texts
            
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            raise
    
    def generate_negative_prompt(
        self,
        positive_prompt: str,
        max_length: int = 80
    ) -> List[str]:
        """生成负面提示词"""
        try:
            # 负面提示词模板
            negative_template = "Generate negative prompts for this positive prompt: "
            full_prompt = negative_template + positive_prompt
            
            # 生成负面提示词
            negative_texts = self.generate_text(
                full_prompt,
                max_length=max_length,
                temperature=0.7,
                num_return_sequences=3
            )
            
            # 清理和格式化
            cleaned_negatives = []
            for text in negative_texts:
                cleaned = text.replace(negative_template, "").strip()
                # 移除正面提示词部分
                if cleaned.startswith(positive_prompt):
                    cleaned = cleaned[len(positive_prompt):].strip()
                # 添加常见的负面词汇
                common_negatives = "blurry, low quality, distorted, ugly, bad anatomy"
                if cleaned:
                    cleaned = cleaned + ", " + common_negatives
                else:
                    cleaned = common_negatives
                cleaned_negatives.append(cleaned)
            
            return cleaned_negatives
            
        except Exception as e:
            logger.error(f"Negative prompt generation failed: {e}")
            raise
    
    def analyze_prompt_quality(
        self,
        prompt: str
    ) -> Dict[str, Any]:
        """分析提示词质量"""
        try:
            analysis = {
                "length": len(prompt),
                "word_count": len(prompt.split()),
                "has_style": False,
                "has_quality": False,
                "has_lighting": False,
                "has_composition": False,
                "suggestions": []
            }
            
            prompt_lower = prompt.lower()
            
            # 检查风格词汇
            style_keywords = ["anime", "realistic", "cartoon", "oil painting", "watercolor", "digital art"]
            analysis["has_style"] = any(keyword in prompt_lower for keyword in style_keywords)
            
            # 检查质量词汇
            quality_keywords = ["high quality", "detailed", "sharp", "4k", "8k", "hd", "professional"]
            analysis["has_quality"] = any(keyword in prompt_lower for keyword in quality_keywords)
            
            # 检查光照词汇
            lighting_keywords = ["lighting", "light", "shadow", "bright", "dark", "golden hour", "dramatic"]
            analysis["has_lighting"] = any(keyword in prompt_lower for keyword in lighting_keywords)
            
            # 检查构图词汇
            composition_keywords = ["composition", "angle", "perspective", "close-up", "wide shot", "portrait"]
            analysis["has_composition"] = any(keyword in prompt_lower for keyword in composition_keywords)
            
            # 生成建议
            if not analysis["has_style"]:
                analysis["suggestions"].append("Add style descriptors (e.g., 'anime style', 'realistic')")
            if not analysis["has_quality"]:
                analysis["suggestions"].append("Add quality descriptors (e.g., 'high quality', 'detailed')")
            if not analysis["has_lighting"]:
                analysis["suggestions"].append("Add lighting descriptors (e.g., 'dramatic lighting', 'soft light')")
            if not analysis["has_composition"]:
                analysis["suggestions"].append("Add composition descriptors (e.g., 'close-up', 'wide angle')")
            
            if analysis["length"] < 20:
                analysis["suggestions"].append("Prompt is too short, consider adding more details")
            elif analysis["length"] > 200:
                analysis["suggestions"].append("Prompt is very long, consider making it more concise")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Prompt analysis failed: {e}")
            raise
    
    def batch_process_prompts(
        self,
        prompts: List[str],
        operation: str = "expand",
        **kwargs
    ) -> Dict[str, List[str]]:
        """批量处理提示词"""
        try:
            results = {}
            
            for i, prompt in enumerate(prompts):
                try:
                    if operation == "expand":
                        result = self.expand_prompt(prompt, **kwargs)
                    elif operation == "optimize":
                        result = self.optimize_prompt(prompt, **kwargs)
                    elif operation == "negative":
                        result = self.generate_negative_prompt(prompt, **kwargs)
                    else:
                        result = [prompt]  # 默认返回原提示词
                    
                    results[f"prompt_{i}"] = result
                    
                except Exception as e:
                    logger.error(f"Failed to process prompt {i}: {e}")
                    results[f"prompt_{i}"] = [prompt]  # 失败时返回原提示词
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def clear_cache(self):
        """清理缓存"""
        device_manager.clear_cache()
