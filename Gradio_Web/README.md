# 🎨 AI创作工作台

**集成文生图、图生图、视频字幕、提示词助理的一体化Gradio平台**

## ✨ 功能特性

### 🎨 图像生成
- **文生图**: 基于文本描述生成高质量图片
- **图生图**: 基于输入图片和文本进行风格转换
- **SDXL支持**: 支持最新的Stable Diffusion XL模型
- **ControlNet**: 支持姿态、深度、边缘控制
- **LoRA微调**: 支持风格和角色微调

### 🎬 视频处理
- **视频字幕**: 自动生成视频字幕
- **多语言支持**: 支持中英日韩等多语言
- **格式导出**: 支持SRT、VTT字幕格式
- **批量处理**: 支持批量视频处理

### 🤖 提示词助理
- **提示词扩写**: 智能扩写和优化提示词
- **风格转换**: 支持多种艺术风格
- **质量分析**: 分析提示词质量并提供建议
- **负面提示词**: 自动生成负面提示词

### 🔧 技术特性
- **多设备支持**: 自动检测GPU/MPS/CPU
- **内存优化**: 智能内存管理和缓存
- **异步处理**: 支持后台任务队列
- **REST API**: 完整的API接口
- **安全过滤**: 内容安全检测

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <your-repo>
cd LLM_Reading_Test/Gradio

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动应用

```bash
# 启动Gradio Web UI
python launch.py ui

# 启动FastAPI后端
python launch.py api

# 同时启动UI和后端
python launch.py both

# 检查依赖
python launch.py ui --check-deps
```

### 3. 访问界面

- **Gradio Web UI**: http://localhost:7860
- **FastAPI后端**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 📁 项目结构

```
Gradio/
├── app.py                 # Gradio主应用
├── launch.py             # 启动脚本
├── requirements.txt      # 依赖列表
├── README.md            # 说明文档
├── configs/             # 配置文件
│   └── settings.py      # 主配置
├── modules/             # 功能模块
│   ├── stable_diffusion_module.py  # SD模块
│   ├── blip_video_module.py        # 视频模块
│   └── nano_gpt_module.py          # GPT模块
├── api/                 # API接口
│   └── fastapi_backend.py          # FastAPI后端
├── utils/               # 工具函数
│   └── device_manager.py           # 设备管理
├── outputs/             # 输出目录
│   ├── images/          # 图片输出
│   ├── videos/          # 视频输出
│   └── text/            # 文本输出
└── cache/               # 缓存目录
```

## 🎯 使用指南

### 文生图

1. 在"文生图"标签页输入提示词
2. 调整参数（尺寸、步数、引导强度等）
3. 点击"生成图片"按钮
4. 查看生成的图片

**示例提示词**:
```
A beautiful sunset over mountains, photorealistic, high resolution, detailed, sharp focus
```

### 图生图

1. 在"图生图"标签页上传输入图片
2. 输入变换提示词
3. 调整变换强度（0.1-1.0）
4. 点击"生成图片"按钮

**示例变换**:
```
transform into anime style, colorful and vibrant
```

### 视频字幕

1. 在"视频字幕"标签页上传视频文件
2. 调整采样参数（帧率、最大帧数）
3. 选择导出格式（SRT/VTT）
4. 点击"生成字幕"按钮

### 提示词助理

1. 在"提示词助理"标签页输入基础提示词
2. 选择操作类型（扩写/优化/负面）
3. 选择风格和参数
4. 点击"处理提示词"按钮

## 🔧 配置说明

### 模型配置

在 `configs/settings.py` 中可以配置：

```python
MODEL_CONFIGS = {
    "stable_diffusion": {
        "default_model": "runwayml/stable-diffusion-v1-5",
        "sdxl_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "controlnet_models": {
            "pose": "lllyasviel/sd-controlnet-openpose",
            "depth": "lllyasviel/sd-controlnet-depth",
            "canny": "lllyasviel/sd-controlnet-canny"
        }
    }
}
```

### 设备配置

```python
DEVICE_CONFIG = {
    "auto_detect": True,
    "prefer_mps": True,  # Mac M系列优先
    "fallback_cpu": True,
    "memory_efficient": True
}
```

### 安全配置

```python
SECURITY_CONFIG = {
    "enable_nsfw_filter": True,
    "enable_watermark": True,
    "max_prompt_length": 500,
    "blocked_keywords": ["explicit", "nsfw", "adult"]
}
```

## 🌐 API接口

### 文生图API

```bash
curl -X POST "http://localhost:8000/api/text2img" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20
  }'
```

### 图生图API

```bash
curl -X POST "http://localhost:8000/api/img2img" \
  -F "prompt=transform into anime style" \
  -F "image=@input.jpg" \
  -F "strength=0.7"
```

### 视频字幕API

```bash
curl -X POST "http://localhost:8000/api/video-caption" \
  -F "video=@input.mp4" \
  -F "fps=1.0" \
  -F "export_srt=true"
```

### 任务状态查询

```bash
curl "http://localhost:8000/api/task/{task_id}"
```

## 🔍 故障排除

### 常见问题

1. **内存不足**
   - 减少 `num_inference_steps`
   - 降低图片尺寸
   - 使用CPU模式

2. **模型加载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间
   - 尝试使用本地模型

3. **生成速度慢**
   - 使用GPU加速
   - 减少推理步数
   - 启用内存优化

4. **Mac M系列问题**
   - 设置 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
   - 使用 `float32` 精度
   - 启用注意力切片

### 日志查看

```bash
# 查看详细日志
python launch.py ui --log-level debug
```

## 🚀 扩展功能

### 计划中的功能

- [ ] ControlNet集成
- [ ] LoRA模型支持
- [ ] 批量处理界面
- [ ] 用户管理系统
- [ ] 模型管理界面
- [ ] 性能监控面板
- [ ] 多语言界面
- [ ] 移动端适配

### 自定义开发

1. **添加新模块**: 在 `modules/` 目录创建新模块
2. **扩展API**: 在 `api/fastapi_backend.py` 添加新接口
3. **修改界面**: 在 `app.py` 中添加新的标签页
4. **配置管理**: 在 `configs/settings.py` 中添加新配置

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请：
1. 查看 [故障排除](#故障排除) 部分
2. 提交 [Issue](https://github.com/your-repo/issues)
3. 联系维护者

---

**享受AI创作的乐趣！** 🎨✨
