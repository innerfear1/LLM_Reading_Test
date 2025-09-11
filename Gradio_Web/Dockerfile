# Gradio工作台Docker配置
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建输出目录
RUN mkdir -p outputs/images outputs/videos outputs/text cache

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 暴露端口
EXPOSE 7860 8000

# 启动命令
CMD ["python", "launch.py", "both"]
