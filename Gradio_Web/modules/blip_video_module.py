"""
BLIP Video模块 - 视频字幕生成、多语言支持、SRT/VTT导出
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any
import logging
from pathlib import Path

from transformers import BlipProcessor, BlipForConditionalGeneration

from utils.device_manager import device_manager
from configs.settings import MODEL_CONFIGS, OUTPUT_DIRS, LANGUAGE_CONFIG

logger = logging.getLogger(__name__)


class BLIPVideoModule:
    """BLIP Video模块"""
    
    def __init__(self):
        self.device = device_manager.get_device()
        self.torch_dtype = device_manager.get_torch_dtype()
        self.processor = None
        self.model = None
        self._load_models()
    
    def _load_models(self):
        """加载BLIP模型"""
        try:
            logger.info("Loading BLIP Video models...")
            
            # 优先使用本地模型
            local_model_path = MODEL_CONFIGS["blip_video"]["local_models"]["blip-video"]
            if local_model_path.exists():
                model_id = str(local_model_path)
                logger.info(f"Using local BLIP model: {model_id}")
            else:
                model_id = MODEL_CONFIGS["blip_video"]["default_model"]
                logger.info(f"Using remote BLIP model: {model_id}")
            
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            
            logger.info("BLIP Video models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP models: {e}")
            raise
    
    def extract_frames(
        self, 
        video_path: str, 
        fps: float = 1.0,
        max_frames: int = 100
    ) -> List[Image.Image]:
        """从视频中提取帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
            
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            raise
    
    def generate_caption(
        self, 
        image: Union[str, Image.Image],
        max_length: int = 100,
        num_beams: int = 5,
        temperature: float = 0.7
    ) -> str:
        """为单张图片生成字幕"""
        try:
            # 处理输入图片
            if isinstance(image, str):
                pil_image = Image.open(image).convert("RGB")
            else:
                pil_image = image
            
            # 预处理
            inputs = self.processor(
                pil_image, 
                return_tensors="pt"
            ).to(self.device)
            
            # 生成字幕
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=True
                )
            
            # 解码输出
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            raise
    
    def generate_video_captions(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 100,
        max_length: int = 100,
        num_beams: int = 5,
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """为视频生成字幕序列"""
        try:
            # 提取帧
            frames = self.extract_frames(video_path, fps, max_frames)
            
            captions = []
            for i, frame in enumerate(frames):
                caption = self.generate_caption(
                    frame, 
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature
                )
                
                timestamp = i / fps
                captions.append({
                    "frame_index": i,
                    "timestamp": timestamp,
                    "caption": caption,
                    "image": frame
                })
                
                logger.info(f"Generated caption for frame {i}: {caption}")
            
            return captions
            
        except Exception as e:
            logger.error(f"Failed to generate video captions: {e}")
            raise
    
    def export_srt(
        self, 
        captions: List[Dict[str, Any]], 
        output_path: str,
        language: str = "en"
    ):
        """导出SRT字幕文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, caption_data in enumerate(captions):
                    start_time = self._format_timestamp(caption_data["timestamp"])
                    end_time = self._format_timestamp(
                        caption_data["timestamp"] + 1.0  # 假设每个字幕显示1秒
                    )
                    
                    f.write(f"{i+1}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{caption_data['caption']}\n\n")
            
            logger.info(f"SRT file exported: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export SRT: {e}")
            raise
    
    def export_vtt(
        self, 
        captions: List[Dict[str, Any]], 
        output_path: str,
        language: str = "en"
    ):
        """导出VTT字幕文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for i, caption_data in enumerate(captions):
                    start_time = self._format_timestamp_vtt(caption_data["timestamp"])
                    end_time = self._format_timestamp_vtt(
                        caption_data["timestamp"] + 1.0
                    )
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{caption_data['caption']}\n\n")
            
            logger.info(f"VTT file exported: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export VTT: {e}")
            raise
    
    def _format_timestamp(self, seconds: float) -> str:
        """格式化时间戳为SRT格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """格式化时间戳为VTT格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def translate_captions(
        self, 
        captions: List[Dict[str, Any]], 
        target_language: str
    ) -> List[Dict[str, Any]]:
        """翻译字幕（需要额外的翻译模型）"""
        # 这里可以集成翻译API或模型
        # 目前返回原始字幕
        logger.warning("Translation not implemented yet")
        return captions
    
    def process_video_with_captions(
        self,
        video_path: str,
        output_dir: str = None,
        fps: float = 1.0,
        max_frames: int = 100,
        export_formats: List[str] = ["srt", "vtt"],
        language: str = "en"
    ) -> Dict[str, Any]:
        """完整的视频字幕处理流程"""
        try:
            if output_dir is None:
                output_dir = OUTPUT_DIRS["videos"]
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成字幕
            captions = self.generate_video_captions(
                video_path, 
                fps=fps, 
                max_frames=max_frames
            )
            
            # 导出字幕文件
            video_name = Path(video_path).stem
            results = {
                "captions": captions,
                "exported_files": []
            }
            
            for format_type in export_formats:
                if format_type == "srt":
                    srt_path = output_dir / f"{video_name}.srt"
                    self.export_srt(captions, str(srt_path), language)
                    results["exported_files"].append(str(srt_path))
                
                elif format_type == "vtt":
                    vtt_path = output_dir / f"{video_name}.vtt"
                    self.export_vtt(captions, str(vtt_path), language)
                    results["exported_files"].append(str(vtt_path))
            
            # 保存JSON格式的字幕数据
            json_path = output_dir / f"{video_name}_captions.json"
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)
            results["exported_files"].append(str(json_path))
            
            logger.info(f"Video processing completed: {len(captions)} captions generated")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            raise
    
    def clear_cache(self):
        """清理缓存"""
        device_manager.clear_cache()
