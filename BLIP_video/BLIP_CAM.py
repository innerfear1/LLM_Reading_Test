import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import logging
import time
from PIL import Image
import sys
import argparse
import os
from threading import Thread, Lock
from queue import Queue

def setup_logging():
    """Configure logging with basic formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class CaptionGenerator:
    """字幕生成器（异步）

    - 持有处理器与模型，并在后台线程中生成字幕
    - 通过队列接收最新帧，避免阻塞摄像头/视频读取
    - 使用锁保护当前字幕的读写
    """
    def __init__(self, processor, model, device):
        """初始化生成器

        参数:
        - processor: BLIP 预处理器
        - model: BLIP 模型
        - device: 推理设备（如 'mps'）
        """
        self.processor = processor
        self.model = model
        self.device = device
        self.current_caption = f"Initializing caption... ({device.upper()})"
        self.caption_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.thread.start()

    def _caption_worker(self):
        """后台线程函数：从队列取帧并更新当前字幕"""
        while self.running:
            try:
                if not self.caption_queue.empty():
                    frame = self.caption_queue.get()
                    caption = self._generate_caption(frame)
                    with self.lock:
                        self.current_caption = caption
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
            time.sleep(0.1)  # Prevent busy waiting

    def _generate_caption(self, image):
        """对单帧生成字幕（同步），供后台线程调用"""
        try:
            # Resize to 640x480 (or any other size)
            image_resized = cv2.resize(image, (640, 480))

            # Convert to RGB
            rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Process the image for captioning
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=5,
                    num_return_sequences=1
                )

            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return f"BLIP: {caption} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return f"BLIP: Caption generation failed ({self.device.upper()})"

    def update_frame(self, frame):
        """将最新帧放入队列（若队列空）以触发后台更新字幕"""
        if self.caption_queue.empty():
            try:
                self.caption_queue.put_nowait(frame.copy())
            except:
                pass  # Queue is full, skip this frame

    def get_caption(self):
        """线程安全地读取当前字幕"""
        with self.lock:
            return self.current_caption

    def stop(self):
        """停止后台线程并回收资源"""
        self.running = False
        self.thread.join()

def get_gpu_usage(device: str = ""):
    """Get accelerator usage info for MPS; otherwise report CPU."""
    try:
        if not device and torch.backends.mps.is_available():
            device = "mps"
        if device == "mps" and torch.backends.mps.is_available():
            # MPS 无法获取详细内存使用情况
            return "MPS (Apple Silicon) active"
        return "CPU mode"
    except Exception:
        return "Accelerator info unavailable"

def load_models(model_path: str = None):
    """Load BLIP model, preferring local path if provided. Device: MPS>CPU.

    参数:
    - model_path: 本地模型目录（含模型与处理器文件）。若为 None 则使用默认仓库名
    """
    try:
        source = model_path if model_path else "Salesforce/blip-image-captioning-large"
        local_only = bool(model_path)

        blip_processor = AutoProcessor.from_pretrained(source, local_files_only=local_only)
        blip_model = AutoModelForImageTextToText.from_pretrained(source, local_files_only=local_only)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        blip_model = blip_model.to(device)

        return blip_processor, blip_model, device
    except Exception as e:
        logging.error(f"Failed to load models: {str(e)}")
        return None, None, None

def live_stream_with_caption(processor, model, device, display_width=1280, display_height=720):
    """摄像头实时字幕与 FPS 显示

    参数:
    - processor/model/device: BLIP 组件与设备
    - display_width/display_height: 预览窗口大小（影响采集分辨率）
    按 'q' 退出
    """
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    logger.info(f"Webcam feed started successfully using {device.upper()}.")
    caption_generator = CaptionGenerator(processor, model, device)

    prev_time = time.time()  # Track time to calculate FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            # Update caption and track FPS
            caption_generator.update_frame(frame)
            current_caption = caption_generator.get_caption()

            # Get accelerator usage info
            gpu_info = get_gpu_usage(device)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Break caption into lines if it overflows
            max_width = 40  # Adjust max width for caption as needed
            caption_lines = [current_caption[i:i + max_width] for i in range(0, len(current_caption), max_width)]

            y_offset = 40
            for line in caption_lines:
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # Display GPU memory usage and FPS
            cv2.putText(frame, gpu_info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            y_offset += 30
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            # Display the video frame
            cv2.imshow("BLIP: Unified Vision-Language Captioning", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user.")
    finally:
        caption_generator.stop()
        cap.release()
        cv2.destroyAllWindows()


def process_video_with_caption(
    processor,
    model,
    device,
    input_path: str,
    output_path: str = None,
    frame_interval: int = 30,
    display: bool = True,
):
    """离线视频字幕生成与可选导出

    - input_path: 输入视频路径
    - output_path: 可选输出带字幕的视频（MP4），为 None 时不保存
    - frame_interval: 每 N 帧重新生成一次字幕，其余复用上次字幕
    - display: 处理时是否显示预览窗口
    按 'q' 退出
    """
    if not os.path.isfile(input_path):
        logger.error(f"Video not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.warning("Failed to open VideoWriter; proceeding without saving.")
            writer = None

    logger.info(
        f"Processing video on {device.upper()} | FPS: {fps:.2f} | Size: {width}x{height} | Frame interval: {frame_interval}"
    )

    last_caption = f"BLIP: Initializing... ({device.upper()})"
    frame_index = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % max(1, frame_interval) == 0:
                try:
                    # Resize and convert color
                    image_resized = cv2.resize(frame, (640, 480))
                    rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)

                    inputs = processor(images=pil_image, return_tensors="pt")
                    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=30,
                            num_beams=5,
                            num_return_sequences=1,
                        )

                    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                    last_caption = f"BLIP: {caption} ({device.upper()})"
                except Exception as e:
                    logging.error(f"Offline caption error at frame {frame_index}: {str(e)}")

            # Draw caption (wrapped)
            max_width = 40
            caption_lines = [last_caption[i:i + max_width] for i in range(0, len(last_caption), max_width)]
            y_offset = 40
            for line in caption_lines:
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # Accelerator info and approx processing FPS
            elapsed = time.time() - start_time
            proc_fps = (frame_index + 1) / elapsed if elapsed > 0 else 0.0
            cv2.putText(frame, get_gpu_usage(device), (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            y_offset += 30
            cv2.putText(frame, f"Proc-FPS: {proc_fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            if display:
                cv2.imshow("BLIP: Offline Captioning", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if writer is not None:
                writer.write(frame)

            frame_index += 1
    except KeyboardInterrupt:
        logger.info("Offline processing interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP live and offline captioning")
    parser.add_argument("--model-path", type=str, default=None, help="Local path to BLIP model directory (already downloaded)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video for offline captioning")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save annotated output video (mp4)")
    parser.add_argument("--frame-interval", type=int, default=30, help="Generate a new caption every N frames (offline mode)")
    parser.add_argument("--no-display", action="store_true", help="Do not display a preview window")
    args = parser.parse_args()

    logger = setup_logging()

    logger.info("Loading BLIP model...")
    blip_processor, blip_model, device = load_models(args.model_path)
    if None in (blip_processor, blip_model):
        logging.error("Failed to load the BLIP model. Exiting.")
        sys.exit(1)

    logger.info(f"Using {device.upper()} for inference.")

    if args.video:
        logger.info("Starting offline video captioning...")
        process_video_with_caption(
            blip_processor,
            blip_model,
            device,
            input_path=args.video,
            output_path=args.output,
            frame_interval=max(1, args.frame_interval),
            display=not args.no_display,
        )
    else:
        logger.info("Starting live stream with BLIP captioning and FPS display...")
        live_stream_with_caption(blip_processor, blip_model, device)

