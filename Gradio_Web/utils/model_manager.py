"""
模型管理器 - 处理模型下载、缓存和本地存储
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from configs.settings import MODEL_CONFIGS, LOCAL_MODEL_DIRS

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.model_configs = MODEL_CONFIGS
        self.local_dirs = LOCAL_MODEL_DIRS
    
    def download_model(
        self, 
        model_id: str, 
        model_type: str,
        local_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False
    ) -> Path:
        """下载模型到本地"""
        try:
            if local_dir is None:
                # 使用默认本地目录
                if model_type in self.local_dirs:
                    local_dir = self.local_dirs[model_type] / "checkpoints" / model_id.split("/")[-1]
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {model_id} to {local_dir}")
            
            # 使用snapshot_download下载完整模型
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                force_download=force_download,
                resume_download=True
            )
            
            logger.info(f"Model downloaded successfully: {downloaded_path}")
            return Path(downloaded_path)
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise
    
    def download_single_file(
        self,
        repo_id: str,
        filename: str,
        model_type: str,
        local_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """下载单个模型文件"""
        try:
            if local_dir is None:
                if model_type in self.local_dirs:
                    local_dir = self.local_dirs[model_type] / "checkpoints"
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {filename} from {repo_id}")
            
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                resume_download=True
            )
            
            logger.info(f"File downloaded successfully: {file_path}")
            return Path(file_path)
            
        except Exception as e:
            logger.error(f"Failed to download file {filename} from {repo_id}: {e}")
            raise
    
    def get_local_model_path(
        self, 
        model_id: str, 
        model_type: str
    ) -> Optional[Path]:
        """获取本地模型路径"""
        try:
            if model_type not in self.local_dirs:
                return None
            
            # 尝试从本地目录查找
            model_name = model_id.split("/")[-1]
            local_path = self.local_dirs[model_type] / "checkpoints" / model_name
            
            if local_path.exists():
                return local_path
            
            # 检查是否有其他变体
            checkpoints_dir = self.local_dirs[model_type] / "checkpoints"
            if checkpoints_dir.exists():
                for item in checkpoints_dir.iterdir():
                    if item.is_dir() and model_name in item.name:
                        return item
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get local model path for {model_id}: {e}")
            return None
    
    def list_local_models(self, model_type: str) -> List[Dict[str, str]]:
        """列出本地模型"""
        try:
            if model_type not in self.local_dirs:
                return []
            
            checkpoints_dir = self.local_dirs[model_type] / "checkpoints"
            if not checkpoints_dir.exists():
                return []
            
            models = []
            for model_dir in checkpoints_dir.iterdir():
                if model_dir.is_dir():
                    models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "size": self._get_directory_size(model_dir)
                    })
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list local models for {model_type}: {e}")
            return []
    
    def remove_model(self, model_path: Union[str, Path]) -> bool:
        """删除本地模型"""
        try:
            model_path = Path(model_path)
            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info(f"Model removed: {model_path}")
                return True
            else:
                logger.warning(f"Model not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove model {model_path}: {e}")
            return False
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, any]:
        """获取模型信息"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return {}
            
            info = {
                "path": str(model_path),
                "name": model_path.name,
                "size": self._get_directory_size(model_path),
                "files": []
            }
            
            # 列出模型文件
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    info["files"].append({
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "path": str(file_path.relative_to(model_path))
                    })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_path}: {e}")
            return {}
    
    def _get_directory_size(self, directory: Path) -> int:
        """计算目录大小"""
        try:
            total_size = 0
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except:
            return 0
    
    def format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def setup_default_models(self) -> Dict[str, bool]:
        """设置默认模型"""
        results = {}
        
        # Stable Diffusion模型
        try:
            sd_path = self.download_model(
                "runwayml/stable-diffusion-v1-5",
                "stable_diffusion"
            )
            results["stable_diffusion"] = True
        except Exception as e:
            logger.error(f"Failed to setup Stable Diffusion model: {e}")
            results["stable_diffusion"] = False
        
        # BLIP Video模型
        try:
            blip_path = self.download_model(
                "Salesforce/blip-video-captioning-base",
                "blip_video"
            )
            results["blip_video"] = True
        except Exception as e:
            logger.error(f"Failed to setup BLIP Video model: {e}")
            results["blip_video"] = False
        
        # GPT-2模型
        try:
            gpt_path = self.download_model(
                "gpt2",
                "nano_gpt"
            )
            results["nano_gpt"] = True
        except Exception as e:
            logger.error(f"Failed to setup GPT-2 model: {e}")
            results["nano_gpt"] = False
        
        return results
    
    def cleanup_cache(self) -> bool:
        """清理缓存"""
        try:
            cache_dirs = [local_dir / "cache" for local_dir in self.local_dirs.values()]
            
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Cache cleaned: {cache_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            return False


# 全局模型管理器实例
model_manager = ModelManager()
