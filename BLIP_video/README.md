# BLIP CAM: Self Hosted Live Image Captioning with Real-Time/Offline Video 🎥

This module provides image captioning using the BLIP (Bootstrapped Language-Image Pretraining) model. It supports real-time webcam streaming and offline video processing on Apple Silicon (M-series, MPS), with captions overlaid and basic performance metrics.

## 🚀 Features

- **Real-Time Video**: Webcam feed capture with overlaid captions
- **Offline Video**: Process existing video files, preview and optional MP4 export with captions
- **State-of-the-Art Captioning**: Uses `Salesforce/blip-image-captioning-large`
- **Apple Silicon Acceleration**: Runs on MPS (M-series)
- **Performance Monitoring**: Display FPS and accelerator status
- **Optimized Architecture**: Background thread for smooth streaming and caption generation

## 📋 Requirements

- Python 3.8+
- Apple Silicon (M-series) recommended for MPS acceleration; CPU 亦可运行
- Webcam（仅实时模式需要）

### Core Dependencies
```
opencv-python>=4.5.0
torch>=1.9.0
transformers>=4.21.0
Pillow>=8.0.0
```

## 🛠️ Installation

在本项目根目录下执行：

```bash
cd BLIP_video
pip install -r requirements.txt
```

运行（实时摄像头）：

```bash
# 在项目根目录
python BLIP_video/BLIP_CAM.py

# 或在模块目录
cd BLIP_video && python BLIP_CAM.py
```

离线视频处理示例：

```bash
python BLIP_video/BLIP_CAM.py --video /path/to/input.mp4 --output /path/to/output.mp4
python BLIP_video/BLIP_CAM.py --video /path/to/input.mp4 --frame-interval 10 --no-display
```

使用本地已下载模型（不联网下载）：

```bash
python BLIP_video/BLIP_CAM.py --model-path /path/to/local/blip-image-captioning-large

# 离线视频 + 本地模型
python BLIP_video/BLIP_CAM.py \
  --model-path /path/to/local/blip-image-captioning-large \
  --video /path/to/input.mp4 \
  --output /path/to/output.mp4
```

## 💡 Use Cases

- **Accessibility Tools**: Real-time scene description for visually impaired users
- **Content Analysis**: Automated video content understanding and tagging
- **Smart Conferencing**: Enhanced video calls with automatic scene descriptions
- **Educational Tools**: Visual learning assistance and scene comprehension
- **Security Systems**: Intelligent surveillance with scene description capabilities

## 🎮 Usage Controls

- Press `Q` to quit the application

## 🔧 CLI Options

程序参数：

- `--video`: 离线模式输入视频路径（不传则开启实时摄像头）
- `--output`: 可选，导出带字幕的 MP4
- `--frame-interval`: 离线模式下每 N 帧重新生成一次字幕（默认 30）
- `--no-display`: 不显示预览窗口，仅处理/导出
- `--model-path`: 指定本地模型目录（已下载的 `Salesforce/blip-image-captioning-large`）


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Salesforce for the BLIP model
- PyTorch team for the deep learning framework
- Hugging Face for the transformers library
- 参考项目：[zawawiAI/BLIP_CAM](https://github.com/zawawiAI/BLIP_CAM.git)

## 📧 Contact

For questions and support, please open an issue in the GitHub repository or reach out to the maintainers.

---
⭐ If you find this project useful, please consider giving it a star!

