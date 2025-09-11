#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理脚本 - 下载、管理和维护模型
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.model_manager import model_manager
from configs.settings import MODEL_CONFIGS, LOCAL_MODEL_DIRS


def list_models():
    """列出所有模型"""
    print("📋 模型列表")
    print("=" * 50)
    
    for model_type, config in MODEL_CONFIGS.items():
        print(f"\n🔧 {model_type.upper()}")
        print("-" * 30)
        
        # 显示配置的模型
        if "default_model" in config:
            print(f"默认模型: {config['default_model']}")
        
        if "sdxl_model" in config:
            print(f"SDXL模型: {config['sdxl_model']}")
        
        if "controlnet_models" in config:
            print("ControlNet模型:")
            for name, model_id in config["controlnet_models"].items():
                print(f"  - {name}: {model_id}")
        
        # 显示本地模型
        local_models = model_manager.list_local_models(model_type)
        if local_models:
            print(f"\n本地模型 ({len(local_models)} 个):")
            for model in local_models:
                size_str = model_manager.format_size(model["size"])
                print(f"  ✅ {model['name']} ({size_str})")
        else:
            print("\n本地模型: 无")


def download_model(model_id: str, model_type: str, force: bool = False):
    """下载模型"""
    print(f"📥 下载模型: {model_id}")
    print("=" * 50)
    
    try:
        local_path = model_manager.download_model(
            model_id=model_id,
            model_type=model_type,
            force_download=force
        )
        
        model_info = model_manager.get_model_info(local_path)
        size_str = model_manager.format_size(model_info["size"])
        
        print(f"✅ 下载成功!")
        print(f"📁 路径: {local_path}")
        print(f"📊 大小: {size_str}")
        print(f"📄 文件数: {len(model_info['files'])}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")


def download_default_models():
    """下载默认模型"""
    print("🚀 下载默认模型")
    print("=" * 50)
    
    results = model_manager.setup_default_models()
    
    print("\n📊 下载结果:")
    for model_type, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {status} {model_type}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n🎯 总体结果: {success_count}/{total_count} 个模型下载成功")


def remove_model(model_path: str):
    """删除模型"""
    print(f"🗑️ 删除模型: {model_path}")
    print("=" * 50)
    
    try:
        success = model_manager.remove_model(model_path)
        if success:
            print("✅ 删除成功!")
        else:
            print("❌ 删除失败或模型不存在")
    except Exception as e:
        print(f"❌ 删除失败: {e}")


def show_model_info(model_path: str):
    """显示模型信息"""
    print(f"📊 模型信息: {model_path}")
    print("=" * 50)
    
    try:
        model_info = model_manager.get_model_info(model_path)
        
        if not model_info:
            print("❌ 模型不存在或无法读取")
            return
        
        print(f"📁 路径: {model_info['path']}")
        print(f"📛 名称: {model_info['name']}")
        print(f"📊 大小: {model_manager.format_size(model_info['size'])}")
        print(f"📄 文件数: {len(model_info['files'])}")
        
        if model_info['files']:
            print(f"\n📄 文件列表:")
            for file_info in model_info['files'][:10]:  # 只显示前10个文件
                size_str = model_manager.format_size(file_info['size'])
                print(f"  - {file_info['name']} ({size_str})")
            
            if len(model_info['files']) > 10:
                print(f"  ... 还有 {len(model_info['files']) - 10} 个文件")
    
    except Exception as e:
        print(f"❌ 获取模型信息失败: {e}")


def cleanup_cache():
    """清理缓存"""
    print("🧹 清理缓存")
    print("=" * 50)
    
    try:
        success = model_manager.cleanup_cache()
        if success:
            print("✅ 缓存清理成功!")
        else:
            print("❌ 缓存清理失败")
    except Exception as e:
        print(f"❌ 缓存清理失败: {e}")


def show_disk_usage():
    """显示磁盘使用情况"""
    print("💾 磁盘使用情况")
    print("=" * 50)
    
    total_size = 0
    
    for model_type, local_dir in LOCAL_MODEL_DIRS.items():
        if local_dir.exists():
            size = model_manager._get_directory_size(local_dir)
            total_size += size
            size_str = model_manager.format_size(size)
            print(f"📁 {model_type}: {size_str}")
        else:
            print(f"📁 {model_type}: 目录不存在")
    
    total_str = model_manager.format_size(total_size)
    print(f"\n💾 总使用量: {total_str}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 列出模型
    subparsers.add_parser("list", help="列出所有模型")
    
    # 下载模型
    download_parser = subparsers.add_parser("download", help="下载模型")
    download_parser.add_argument("model_id", help="模型ID")
    download_parser.add_argument("model_type", help="模型类型")
    download_parser.add_argument("--force", action="store_true", help="强制重新下载")
    
    # 下载默认模型
    subparsers.add_parser("download-default", help="下载默认模型")
    
    # 删除模型
    remove_parser = subparsers.add_parser("remove", help="删除模型")
    remove_parser.add_argument("model_path", help="模型路径")
    
    # 显示模型信息
    info_parser = subparsers.add_parser("info", help="显示模型信息")
    info_parser.add_argument("model_path", help="模型路径")
    
    # 清理缓存
    subparsers.add_parser("cleanup", help="清理缓存")
    
    # 显示磁盘使用
    subparsers.add_parser("disk-usage", help="显示磁盘使用情况")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "list":
            list_models()
        elif args.command == "download":
            download_model(args.model_id, args.model_type, args.force)
        elif args.command == "download-default":
            download_default_models()
        elif args.command == "remove":
            remove_model(args.model_path)
        elif args.command == "info":
            show_model_info(args.model_path)
        elif args.command == "cleanup":
            cleanup_cache()
        elif args.command == "disk-usage":
            show_disk_usage()
    
    except KeyboardInterrupt:
        print("\n\n👋 操作被用户中断")
    except Exception as e:
        print(f"\n❌ 操作失败: {e}")


if __name__ == "__main__":
    main()
