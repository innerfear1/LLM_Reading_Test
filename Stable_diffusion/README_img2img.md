# 图生图使用指南 (Image-to-Image Guide)

## 🎨 什么是图生图？

图生图 (Image-to-Image) 是 Stable Diffusion 的一个重要功能，它可以根据一张输入图片和文字描述，生成新的图片。这个功能可以用于：

- 🎭 **风格转换**: 将照片转换为不同艺术风格
- 🌟 **图像增强**: 改善图片质量或添加特效
- 🔄 **图像编辑**: 根据文字描述修改图片内容
- 🎨 **创意变换**: 将现实图片转换为奇幻或科幻风格

## 📋 参数说明

### 核心参数

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `prompt` | str | - | 描述你想要的变换效果 |
| `image` | str/PIL.Image | - | 输入图片路径或PIL图片对象 |
| `strength` | float | 0.0-1.0 | 变换强度，越高变化越大 |
| `guidance_scale` | float | 1.0-20.0 | 提示词遵循程度 |
| `num_inference_steps` | int | 10-100 | 推理步数，越多质量越好但越慢 |

### Strength 参数详解

- **0.0-0.3**: 轻微调整，保持原图大部分特征
- **0.4-0.6**: 中等变换，平衡原图和提示词
- **0.7-0.8**: 较大变换，明显改变风格
- **0.9-1.0**: 完全变换，几乎重新生成

## 🚀 使用方法

### 方法1: 直接运行主程序

```bash
cd Stable_diffusion
python Laten_SD.py
```

选择选项 `2` 或 `3` 进行图生图。

### 方法2: 使用示例脚本

```bash
cd Stable_diffusion
python img2img_example.py
```

### 方法3: 在代码中使用

```python
from Laten_SD import StableDiffusionApp

# 初始化应用
app = StableDiffusionApp()

# 图生图
result = app.generate_img2img(
    prompt="transform into anime style",
    image="path/to/your/image.jpg",
    strength=0.7,
    guidance_scale=7.5,
    num_inference_steps=30
)

# 显示结果
app.display_images(result, "Anime Style")
```

## 💡 使用技巧

### 1. 提示词技巧

**好的提示词示例:**
- `"transform into anime style, colorful and vibrant"`
- `"make it look like a watercolor painting with soft colors"`
- `"add dramatic lighting and cinematic atmosphere"`

**避免的提示词:**
- 过于模糊: `"make it better"`
- 过于复杂: `"transform into anime style with cyberpunk elements and vintage colors and modern lighting"`

### 2. Strength 选择指南

| 变换类型 | 推荐 Strength | 说明 |
|----------|---------------|------|
| 轻微调色 | 0.2-0.4 | 保持原图结构 |
| 风格转换 | 0.5-0.7 | 平衡原图和风格 |
| 大幅变换 | 0.8-0.9 | 显著改变外观 |
| 完全重绘 | 0.9-1.0 | 几乎重新生成 |

### 3. 图片准备

- **格式**: 支持 JPG, PNG, WEBP 等常见格式
- **尺寸**: 建议 512x512 或 768x768
- **质量**: 输入图片质量越高，输出效果越好
- **内容**: 清晰的图片比模糊的图片效果更好

## 🎯 常见应用场景

### 1. 艺术风格转换

```python
# 转换为油画风格
result = app.generate_img2img(
    prompt="make it look like a classical oil painting with rich colors",
    image="portrait.jpg",
    strength=0.8
)
```

### 2. 动漫化

```python
# 转换为动漫风格
result = app.generate_img2img(
    prompt="transform into anime style, colorful and vibrant",
    image="photo.jpg",
    strength=0.7
)
```

### 3. 环境变换

```python
# 改变背景环境
result = app.generate_img2img(
    prompt="change background to a beautiful garden with flowers",
    image="person.jpg",
    strength=0.6
)
```

### 4. 光照效果

```python
# 添加戏剧性光照
result = app.generate_img2img(
    prompt="add dramatic golden hour lighting",
    image="landscape.jpg",
    strength=0.5
)
```

## ⚠️ 注意事项

1. **内存使用**: 图生图比文生图消耗更多内存
2. **处理时间**: 推理步数越多，处理时间越长
3. **图片质量**: 输入图片质量直接影响输出效果
4. **参数调优**: 不同图片可能需要不同的参数设置

## 🔧 故障排除

### 问题1: 生成全黑图片
- 检查输入图片是否正确加载
- 尝试降低 `strength` 值
- 确保提示词清晰明确

### 问题2: 变化太小
- 增加 `strength` 值 (0.7-0.9)
- 使用更具体的提示词
- 增加 `guidance_scale` 值

### 问题3: 变化太大
- 降低 `strength` 值 (0.3-0.6)
- 使用更温和的提示词
- 增加 `num_inference_steps` 提高质量

### 问题4: 内存不足
- 减少 `num_inference_steps`
- 使用较小的输入图片
- 在 MPS 设备上使用 `float32` 精度

## 📁 输出文件

生成的图片会保存在 `./outputs/` 目录下，文件名格式为：
- `img2img_[prompt]_[number].png`

例如: `img2img_anime_style_1.png`

## 🎉 开始使用

现在你可以开始使用图生图功能了！建议从简单的风格转换开始，逐步尝试更复杂的变换。

```bash
# 运行示例
python img2img_example.py
```

选择选项 `1` 开始简单示例，或者选择 `2` 使用你自己的图片进行自定义变换。
