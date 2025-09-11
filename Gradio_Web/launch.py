#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio工作台启动脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.settings import API_CONFIG, OUTPUT_DIRS

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def launch_gradio_ui():
    """启动Gradio Web UI"""
    try:
        from app import main
        logger.info("🚀 Starting Gradio Web UI...")
        main()
    except Exception as e:
        logger.error(f"Failed to start Gradio UI: {e}")
        raise


def launch_fastapi_backend():
    """启动FastAPI后端"""
    try:
        import uvicorn
        from api.fastapi_backend import app
        
        logger.info("🚀 Starting FastAPI Backend...")
        uvicorn.run(
            app,
            host=API_CONFIG["host"],
            port=8000,  # 使用不同端口避免冲突
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start FastAPI backend: {e}")
        raise


def check_dependencies():
    """检查依赖"""
    required_packages = [
        "gradio", "fastapi", "uvicorn", "torch", "diffusers", 
        "transformers", "PIL", "opencv-python", "moviepy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def setup_environment():
    """设置环境"""
    # 创建必要的目录
    for output_dir in OUTPUT_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")
    
    # 设置环境变量
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Mac M系列优化
    
    logger.info("✅ Environment setup completed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI创作工作台启动器")
    parser.add_argument(
        "mode",
        choices=["ui", "api", "both"],
        help="启动模式: ui (Gradio界面), api (FastAPI后端), both (同时启动)"
    )
    parser.add_argument(
        "--host",
        default=API_CONFIG["host"],
        help="服务器地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=API_CONFIG["port"],
        help="服务器端口"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="检查依赖"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check_deps or not check_dependencies():
        if not check_dependencies():
            sys.exit(1)
        logger.info("✅ All dependencies are installed")
        return
    
    # 设置环境
    setup_environment()
    
    # 根据模式启动
    if args.mode == "ui":
        launch_gradio_ui()
    elif args.mode == "api":
        launch_fastapi_backend()
    elif args.mode == "both":
        import threading
        import time
        
        # 启动FastAPI后端
        api_thread = threading.Thread(target=launch_fastapi_backend)
        api_thread.daemon = True
        api_thread.start()
        
        # 等待后端启动
        time.sleep(3)
        
        # 启动Gradio UI
        launch_gradio_ui()


if __name__ == "__main__":
    main()
