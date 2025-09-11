"""
设备管理器 - 统一管理GPU/CPU/MPS设备
"""

import torch
import logging
from typing import Optional, Dict, Any
from configs.settings import DEVICE_CONFIG

logger = logging.getLogger(__name__)


class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.device = self._detect_device()
        self.device_info = self._get_device_info()
        logger.info(f"Device detected: {self.device}")
        logger.info(f"Device info: {self.device_info}")
    
    def _detect_device(self) -> str:
        """自动检测最佳设备"""
        if not DEVICE_CONFIG["auto_detect"]:
            return "cpu"
        
        # 优先使用MPS (Mac M系列)
        if DEVICE_CONFIG["prefer_mps"] and torch.backends.mps.is_available():
            return "mps"
        
        # 其次使用CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        # 最后使用CPU
        if DEVICE_CONFIG["fallback_cpu"]:
            return "cpu"
        
        raise RuntimeError("No suitable device found")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            "device": self.device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available()
        }
        
        if self.device == "cuda":
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory
            })
        elif self.device == "mps":
            info.update({
                "mps_version": "1.0"  # MPS版本信息
            })
        
        return info
    
    def get_device(self) -> str:
        """获取当前设备"""
        return self.device
    
    def get_torch_dtype(self) -> torch.dtype:
        """根据设备获取合适的torch数据类型"""
        if self.device == "mps":
            # MPS设备使用float32避免精度问题
            return torch.float32
        elif self.device == "cuda":
            # CUDA设备可以使用float16提高性能
            return torch.float16
        else:
            # CPU设备使用float32
            return torch.float32
    
    def optimize_for_device(self, model: Any) -> Any:
        """为当前设备优化模型"""
        if self.device == "mps":
            # MPS优化
            if hasattr(model, 'enable_attention_slicing'):
                model.enable_attention_slicing()
            if hasattr(model, 'disable_memory_efficient_attention'):
                try:
                    model.disable_memory_efficient_attention()
                except:
                    pass
        elif self.device == "cuda":
            # CUDA优化
            if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                try:
                    model.enable_xformers_memory_efficient_attention()
                except:
                    pass
        
        return model
    
    def clear_cache(self):
        """清理设备缓存"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        info = {}
        
        if self.device == "cuda":
            info = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "max_reserved": torch.cuda.max_memory_reserved()
            }
        elif self.device == "mps":
            # MPS内存信息有限
            info = {
                "device": "mps",
                "note": "MPS memory info not available"
            }
        else:
            info = {
                "device": "cpu",
                "note": "CPU memory info not available"
            }
        
        return info
    
    def set_memory_fraction(self, fraction: float):
        """设置GPU内存使用比例"""
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(fraction)
        elif self.device == "mps":
            # MPS不支持内存比例设置
            logger.warning("MPS does not support memory fraction setting")
    
    def is_available(self) -> bool:
        """检查设备是否可用"""
        try:
            if self.device == "cuda":
                return torch.cuda.is_available()
            elif self.device == "mps":
                return torch.backends.mps.is_available()
            else:
                return True
        except:
            return False


# 全局设备管理器实例
device_manager = DeviceManager()
