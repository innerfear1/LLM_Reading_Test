# 📁 模型文件路径指南

## 🗂️ 模型文件应该放在哪里？

### 1. **自动下载（推荐）**
模型会在首次使用时自动从HuggingFace下载到缓存目录：
```
~/.cache/huggingface/hub/
```

### 2. **本地存储目录**
如果你想手动管理模型，可以放在以下目录：

```
Gradio/models/
├── stable_diffusion/
│   ├── checkpoints/          # 模型检查点
│   │   ├── sd-v1-5/         # SD 1.5模型
│   │   ├── sdxl/            # SDXL模型
│   │   └── custom/          # 自定义模型
│   ├── cache/               # HuggingFace缓存
│   └── configs/             # 模型配置
├── blip_video/
│   ├── checkpoints/         # BLIP模型
│   ├── cache/              # 缓存
│   └── configs/            # 配置
├── nano_gpt/
│   ├── checkpoints/        # GPT模型
│   ├── cache/             # 缓存
│   └── configs/           # 配置
└── dalle_mini/
    ├── checkpoints/       # DALLE模型
    ├── cache/            # 缓存
    └── configs/          # 配置
```

## 🚀 快速设置

### 方法1：使用模型管理工具（推荐）
```bash
# 查看所有模型
python manage_models.py list

# 下载默认模型
python manage_models.py download-default

# 下载特定模型
python manage_models.py download runwayml/stable-diffusion-v1-5 stable_diffusion

# 查看磁盘使用
python manage_models.py disk-usage
```

### 方法2：手动下载
```bash
# 使用huggingface-cli
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./models/stable_diffusion/checkpoints/sd-v1-5

# 使用git lfs
git lfs clone https://huggingface.co/runwayml/stable-diffusion-v1-5 ./models/stable_diffusion/checkpoints/sd-v1-5
```

### 方法3：直接复制
如果你已经有模型文件，直接复制到对应目录：
```bash
# 复制SD模型
cp -r /path/to/your/sd-model ./models/stable_diffusion/checkpoints/sd-v1-5/

# 复制BLIP模型
cp -r /path/to/your/blip-model ./models/blip_video/checkpoints/blip-video/
```

## 📋 具体模型路径

### Stable Diffusion模型
- **SD 1.5**: `./models/stable_diffusion/checkpoints/sd-v1-5/`
- **SDXL**: `./models/stable_diffusion/checkpoints/sdxl/`
- **ControlNet**: `./models/stable_diffusion/checkpoints/controlnet/`
- **LoRA**: `./models/stable_diffusion/checkpoints/lora/`

### BLIP Video模型
- **基础模型**: `./models/blip_video/checkpoints/blip-video/`
- **多语言模型**: `./models/blip_video/checkpoints/blip-video-multilang/`

### Nano GPT模型
- **GPT-2**: `./models/nano_gpt/checkpoints/gpt2/`
- **自定义模型**: `./models/nano_gpt/checkpoints/custom/`

### DALLE Mini模型
- **基础模型**: `./models/dalle_mini/checkpoints/dalle-mini/`

## ⚙️ 环境变量设置

### 设置HuggingFace缓存目录
```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export HF_HOME=./models/cache
export TRANSFORMERS_CACHE=./models/cache
export HF_HUB_CACHE=./models/cache
```

### 在Python中设置
```python
import os
os.environ["HF_HOME"] = "./models/cache"
os.environ["TRANSFORMERS_CACHE"] = "./models/cache"
```

## 🔧 配置文件修改

如果你想使用本地模型，可以修改 `configs/settings.py`：

```python
MODEL_CONFIGS = {
    "stable_diffusion": {
        "default_model": "./models/stable_diffusion/checkpoints/sd-v1-5",  # 本地路径
        # 或者
        "default_model": "runwayml/stable-diffusion-v1-5",  # HuggingFace ID
    }
}
```

## 💾 存储空间需求

### 模型大小估算
- **SD 1.5**: ~4GB
- **SDXL**: ~7GB
- **ControlNet**: ~1.5GB each
- **BLIP Video**: ~1GB
- **GPT-2**: ~500MB
- **DALLE Mini**: ~1GB

### 总存储需求
- **最小配置**: ~8GB (基础模型)
- **完整配置**: ~20GB (所有模型)
- **推荐配置**: ~50GB (包含LoRA和自定义模型)

## 🔍 故障排除

### 模型下载失败
1. **检查网络连接**
2. **使用镜像站点**:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. **手动下载**: 使用浏览器下载后解压到对应目录

### 模型加载错误
1. **检查文件完整性**: 确保所有必需文件都存在
2. **检查权限**: 确保有读取权限
3. **检查路径**: 确保路径正确

### 内存不足
1. **使用CPU模式**: 在配置中设置 `device="cpu"`
2. **启用量化**: 使用 `torch_dtype=torch.float16`
3. **减少批处理**: 设置 `num_images_per_prompt=1`

## 📝 最佳实践

1. **使用模型管理工具**: 统一管理所有模型
2. **定期清理缓存**: 释放磁盘空间
3. **备份重要模型**: 避免重新下载
4. **监控磁盘使用**: 避免空间不足
5. **使用符号链接**: 节省磁盘空间

## 🎯 推荐设置

### 开发环境
```bash
# 下载基础模型
python manage_models.py download-default

# 设置环境变量
export HF_HOME=./models/cache
```

### 生产环境
```bash
# 预下载所有模型
python manage_models.py download-default

# 使用Docker挂载模型目录
docker run -v ./models:/app/models your-image
```

---

**总结**: 模型文件可以放在 `Gradio/models/` 目录下，使用 `python manage_models.py` 工具可以方便地管理所有模型！
