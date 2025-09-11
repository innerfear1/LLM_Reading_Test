#!/bin/bash

# Gradio工作台快速启动脚本

echo "🎨 AI创作工作台快速启动"
echo "=========================="

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 检测到虚拟环境: $VIRTUAL_ENV"
else
    echo "⚠️ 建议在虚拟环境中运行"
fi

# 检查依赖
echo ""
echo "🔍 检查依赖..."
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt 存在"
else
    echo "❌ requirements.txt 不存在"
    exit 1
fi

# 安装依赖
echo ""
echo "📦 安装依赖..."
pip install -r requirements.txt

# 运行测试
echo ""
echo "🧪 运行测试..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 测试通过！"
    echo ""
    echo "🚀 启动选项:"
    echo "1. Gradio Web UI:     python launch.py ui"
    echo "2. FastAPI Backend:   python launch.py api"
    echo "3. 同时启动:          python launch.py both"
    echo "4. Docker部署:        docker-compose up"
    echo ""
    echo "🌐 访问地址:"
    echo "- Gradio UI: http://localhost:7860"
    echo "- API文档:   http://localhost:8000/docs"
    echo ""
    
    # 询问是否立即启动
    read -p "是否立即启动Gradio UI? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 启动Gradio UI..."
        python launch.py ui
    fi
else
    echo "❌ 测试失败，请检查依赖和配置"
    exit 1
fi
