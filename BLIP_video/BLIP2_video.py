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
        """初始化生成器"""
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
            return f"BLIP2: {caption} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return f"BLIP2: Caption generation failed ({self.device.upper()})"

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

def get_gpu_usage(device: str = None):
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
    """Load BLIP2 model, preferring local path if provided. Device: MPS>CPU.

    参数:
    - model_path: 本地模型目录（含模型与处理器文件）。若为 None 则使用默认仓库名
    """
    try:
        # blip2 model path
        source = model_path if model_path else "Salesforce/blip2-opt-2.7b"
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
            cv2.imshow("BLIP2: Unified Vision-Language Captioning", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user.")
    finally:
        caption_generator.stop()
        cap.release()
        cv2.destroyAllWindows()


def live_stream_with_vqa(processor, model, device, question_text: str, display_width=1280, display_height=720, frame_interval: int = 30):
    """摄像头实时 VQA：定期对当前帧进行回答并打印/叠加显示"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    logger.info(f"Webcam feed started successfully using {device.upper()} (VQA). Question: {question_text}")
    vqa = VQA(processor, model, device, question_text=question_text)

    prev_time = time.time()
    frame_index = 0
    last_printed = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            if frame_index % max(1, frame_interval) == 0:
                vqa.update_frame(frame)
                current_answer = vqa.get_answer()
                if current_answer != last_printed:
                    print(current_answer)
                    logger.info(current_answer)
                    last_printed = current_answer
            else:
                current_answer = vqa.get_answer()

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # draw
            max_width = 40
            lines = [current_answer[i:i + max_width] for i in range(0, len(current_answer), max_width)]
            y_offset = 40
            for line in lines:
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            cv2.putText(frame, get_gpu_usage(device), (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            y_offset += 30
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            cv2.imshow("BLIP2: VQA", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1
    except KeyboardInterrupt:
        logger.info("VQA stream interrupted by user.")
    finally:
        vqa.stop()
        cap.release()
        cv2.destroyAllWindows()


def process_video_with_vqa(
    processor,
    model,
    device,
    input_path: str,
    question_text: str,
    output_path: str = None,
    frame_interval: int = 30,
    display: bool = True,
):
    """离线视频 VQA：定期回答问题，打印并可选导出带答案的视频"""
    if not os.path.isfile(input_path):
        logger.error(f"Video not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.warning("Failed to open VideoWriter; proceeding without saving.")
            writer = None

    logger.info(f"Processing VQA on {device.upper()} | Q: {question_text} | Interval: {frame_interval}")

    last_answer = f"VQA: Initializing... ({device.upper()})"
    frame_index = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % max(1, frame_interval) == 0:
                try:
                    image_resized = cv2.resize(frame, (640, 480))
                    rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)

                    inputs = processor(images=pil_image, text=question_text, return_tensors="pt")
                    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=30,
                            num_beams=5,
                            num_return_sequences=1,
                        )

                    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                    last_answer = f"VQA: {answer} ({device.upper()})"
                    print(last_answer)
                    logger.info(last_answer)
                except Exception as e:
                    logging.error(f"Offline VQA error at frame {frame_index}: {str(e)}")

            # draw
            max_width = 40
            lines = [last_answer[i:i + max_width] for i in range(0, len(last_answer), max_width)]
            y_offset = 40
            for line in lines:
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # fps
            elapsed = time.time() - start_time
            proc_fps = (frame_index + 1) / elapsed if elapsed > 0 else 0.0
            cv2.putText(frame, get_gpu_usage(device), (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            y_offset += 30
            cv2.putText(frame, f"Proc-FPS: {proc_fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            if display:
                cv2.imshow("BLIP2: Offline VQA", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if writer is not None:
                writer.write(frame)

            frame_index += 1
    except KeyboardInterrupt:
        logger.info("Offline VQA interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
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

    last_caption = f"BLIP2: Initializing... ({device.upper()})"
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
                    last_caption = f"BLIP2: {caption} ({device.upper()})"
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
                cv2.imshow("BLIP2: Offline Captioning", frame)
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

# vqa Visual question answering (VQA)
class VQA:
    """
    exp:
    # ask a random question.
    question = "Which city is this photo taken?"
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)
    model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
    # ['singapore']
    """
    def __init__(self, processor, model, device, question_text: str = "What is in the image?"):
        self.processor = processor
        self.model = model
        self.device = device
        self.question_text = question_text
        self.current_question = f"VQA: Waiting... ({device.upper()})"
        self.question_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._vqa_worker)
        self.thread.daemon = True
        self.thread.start()

    def _vqa_worker(self):
        """后台线程函数：从队列取帧并更新当前问题"""
        while self.running:
            try:
                if not self.question_queue.empty():
                    frame = self.question_queue.get()
                    question = self._generate_answer(frame)
                    with self.lock:
                        self.current_question = question
            except Exception as e:
                logging.error(f"VQA worker error: {str(e)}")
            time.sleep(0.1)  # Prevent busy waiting
    
    def _generate_answer(self, image):
        """对单帧与给定问题进行作答（同步），供后台线程调用"""
        try:
            # Resize to 640x480 (or any other size)
            image_resized = cv2.resize(image, (640, 480))
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Process the image and the question for answering
            inputs = self.processor(images=pil_image, text=self.question_text, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=5,
                    num_return_sequences=1
                )

            answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return f"VQA: {answer} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"Question generation error: {str(e)}")
            return f"VQA: Answer generation failed ({self.device.upper()})"
        
    def update_frame(self, frame):
        """将最新帧放入队列（若队列空）以触发后台更新问题"""
        if self.question_queue.empty():
            try:
                self.question_queue.put_nowait(frame.copy())
            except:
                pass  # Queue is full, skip this frame
    
    def get_question(self):
        """线程安全地读取当前答案（兼容方法名）"""
        with self.lock:
            return self.current_question
    
    def get_answer(self):
        """线程安全地读取当前答案（推荐）"""
        with self.lock:
            return self.current_question
        
    def stop(self):
        """停止后台线程并回收资源"""
        self.running = False
        self.thread.join()

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP2 live/offline captioning & VQA")
    parser.add_argument("--model-path", type=str, default=None, help="Local path to BLIP2 model directory (already downloaded)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video for offline mode")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save annotated output video (mp4)")
    parser.add_argument("--frame-interval", type=int, default=30, help="Generate a new result every N frames (offline/VQA)")
    parser.add_argument("--no-display", action="store_true", help="Do not display a preview window")
    parser.add_argument("--vqa", action="store_true", help="Enable VQA mode (default off)")
    parser.add_argument("--question", type=str, default="What is in the image?", help="Question to ask in VQA mode")

    args = parser.parse_args()

    logger = setup_logging()

    logger.info("Loading BLIP2 model...")
    blip_processor, blip_model, device = load_models(args.model_path)
    if None in (blip_processor, blip_model):
        logging.error("Failed to load the BLIP2 model. Exiting.")
        sys.exit(1)

    logger.info(f"Using {device.upper()} for inference.")

    if args.vqa:
        if args.video:
            logger.info("Starting offline VQA...")
            process_video_with_vqa(
                blip_processor,
                blip_model,
                device,
                input_path=args.video,
                question_text=args.question,
                output_path=args.output,
                frame_interval=max(1, args.frame_interval),
                display=not args.no_display,
            )
        else:
            logger.info("Starting live VQA stream...")
            live_stream_with_vqa(
                blip_processor,
                blip_model,
                device,
                question_text=args.question,
                frame_interval=max(1, args.frame_interval),
            )
    else:
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
            logger.info("Starting live stream with BLIP2 captioning and FPS display...")
            live_stream_with_caption(blip_processor, blip_model, device)
