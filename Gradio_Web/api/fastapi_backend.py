"""
FastAPI后端服务 - 提供REST API接口
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import uuid
import asyncio
from pathlib import Path
import json

# 导入模块
from modules.stable_diffusion_module import StableDiffusionModule
from modules.blip_video_module import BLIPVideoModule
from modules.nano_gpt_module import NanoGPTModule
from configs.settings import OUTPUT_DIRS, SECURITY_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="AI创作工作台 API",
    description="集成文生图、图生图、视频字幕、提示词助理的API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模块实例
sd_module = None
blip_module = None
gpt_module = None

# 任务队列
task_queue = {}
completed_tasks = {}


# Pydantic模型
class Text2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    use_sdxl: bool = False


class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    strength: float = 0.7
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    use_sdxl: bool = False


class VideoCaptionRequest(BaseModel):
    fps: float = 1.0
    max_frames: int = 100
    max_length: int = 100
    export_srt: bool = True
    export_vtt: bool = True


class PromptRequest(BaseModel):
    base_prompt: str
    operation: str = "expand"  # expand, optimize, negative
    style: str = "detailed"
    max_length: int = 150
    temperature: float = 0.8


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# 初始化模块
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模块"""
    global sd_module, blip_module, gpt_module
    
    try:
        logger.info("Initializing API modules...")
        
        # 初始化模块
        sd_module = StableDiffusionModule()
        blip_module = BLIPVideoModule()
        gpt_module = NanoGPTModule()
        
        logger.info("✅ All API modules initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize API modules: {e}")
        raise


def validate_prompt(prompt: str) -> bool:
    """验证提示词安全性"""
    if not prompt or len(prompt.strip()) == 0:
        return False
    
    if len(prompt) > SECURITY_CONFIG["max_prompt_length"]:
        return False
    
    prompt_lower = prompt.lower()
    for keyword in SECURITY_CONFIG["blocked_keywords"]:
        if keyword in prompt_lower:
            return False
    
    return True


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "modules": {
            "stable_diffusion": sd_module is not None,
            "blip_video": blip_module is not None,
            "nano_gpt": gpt_module is not None
        }
    }


# 文生图API
@app.post("/api/text2img", response_model=TaskResponse)
async def text2img_api(request: Text2ImgRequest, background_tasks: BackgroundTasks):
    """文生图API"""
    try:
        if not validate_prompt(request.prompt):
            raise HTTPException(status_code=400, detail="提示词不符合安全要求")
        
        if sd_module is None:
            raise HTTPException(status_code=500, detail="Stable Diffusion模块未初始化")
        
        # 创建任务
        task_id = str(uuid.uuid4())
        task_queue[task_id] = {
            "type": "text2img",
            "request": request.dict(),
            "status": "pending"
        }
        
        # 添加后台任务
        background_tasks.add_task(process_text2img_task, task_id)
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="任务已创建，正在处理中..."
        )
        
    except Exception as e:
        logger.error(f"Text2Img API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_text2img_task(task_id: str):
    """处理文生图任务"""
    try:
        task = task_queue[task_id]
        task["status"] = "processing"
        
        request_data = task["request"]
        
        # 生成图片
        images = sd_module.text2img(
            prompt=request_data["prompt"],
            negative_prompt=request_data["negative_prompt"],
            width=request_data["width"],
            height=request_data["height"],
            num_inference_steps=request_data["num_inference_steps"],
            guidance_scale=request_data["guidance_scale"],
            num_images_per_prompt=request_data["num_images_per_prompt"],
            seed=request_data["seed"],
            use_sdxl=request_data["use_sdxl"]
        )
        
        # 保存图片
        image_paths = []
        for i, img in enumerate(images):
            image_path = OUTPUT_DIRS["images"] / f"{task_id}_{i}.png"
            img.save(image_path)
            image_paths.append(str(image_path))
        
        # 完成任务
        completed_tasks[task_id] = {
            "status": "completed",
            "result": {
                "images": image_paths,
                "count": len(images)
            }
        }
        
        # 从队列中移除
        del task_queue[task_id]
        
    except Exception as e:
        logger.error(f"Text2Img task {task_id} failed: {e}")
        completed_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }
        if task_id in task_queue:
            del task_queue[task_id]


# 图生图API
@app.post("/api/img2img", response_model=TaskResponse)
async def img2img_api(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.7),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(7.5),
    num_images_per_prompt: int = Form(1),
    seed: Optional[int] = Form(None),
    use_sdxl: bool = Form(False),
    image: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """图生图API"""
    try:
        if not validate_prompt(prompt):
            raise HTTPException(status_code=400, detail="提示词不符合安全要求")
        
        if sd_module is None:
            raise HTTPException(status_code=500, detail="Stable Diffusion模块未初始化")
        
        # 保存上传的图片
        task_id = str(uuid.uuid4())
        image_path = OUTPUT_DIRS["images"] / f"{task_id}_input.png"
        
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # 创建任务
        task_queue[task_id] = {
            "type": "img2img",
            "request": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images_per_prompt,
                "seed": seed,
                "use_sdxl": use_sdxl,
                "image_path": str(image_path)
            },
            "status": "pending"
        }
        
        # 添加后台任务
        background_tasks.add_task(process_img2img_task, task_id)
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="任务已创建，正在处理中..."
        )
        
    except Exception as e:
        logger.error(f"Img2Img API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_img2img_task(task_id: str):
    """处理图生图任务"""
    try:
        task = task_queue[task_id]
        task["status"] = "processing"
        
        request_data = task["request"]
        
        # 生成图片
        images = sd_module.img2img(
            prompt=request_data["prompt"],
            negative_prompt=request_data["negative_prompt"],
            image=request_data["image_path"],
            strength=request_data["strength"],
            num_inference_steps=request_data["num_inference_steps"],
            guidance_scale=request_data["guidance_scale"],
            num_images_per_prompt=request_data["num_images_per_prompt"],
            seed=request_data["seed"],
            use_sdxl=request_data["use_sdxl"]
        )
        
        # 保存图片
        image_paths = []
        for i, img in enumerate(images):
            image_path = OUTPUT_DIRS["images"] / f"{task_id}_{i}.png"
            img.save(image_path)
            image_paths.append(str(image_path))
        
        # 完成任务
        completed_tasks[task_id] = {
            "status": "completed",
            "result": {
                "images": image_paths,
                "count": len(images)
            }
        }
        
        # 从队列中移除
        del task_queue[task_id]
        
    except Exception as e:
        logger.error(f"Img2Img task {task_id} failed: {e}")
        completed_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }
        if task_id in task_queue:
            del task_queue[task_id]


# 视频字幕API
@app.post("/api/video-caption", response_model=TaskResponse)
async def video_caption_api(
    video: UploadFile = File(...),
    fps: float = Form(1.0),
    max_frames: int = Form(100),
    max_length: int = Form(100),
    export_srt: bool = Form(True),
    export_vtt: bool = Form(True),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """视频字幕API"""
    try:
        if blip_module is None:
            raise HTTPException(status_code=500, detail="BLIP Video模块未初始化")
        
        # 保存上传的视频
        task_id = str(uuid.uuid4())
        video_path = OUTPUT_DIRS["videos"] / f"{task_id}_input.mp4"
        
        with open(video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # 创建任务
        task_queue[task_id] = {
            "type": "video_caption",
            "request": {
                "video_path": str(video_path),
                "fps": fps,
                "max_frames": max_frames,
                "max_length": max_length,
                "export_srt": export_srt,
                "export_vtt": export_vtt
            },
            "status": "pending"
        }
        
        # 添加后台任务
        background_tasks.add_task(process_video_caption_task, task_id)
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="任务已创建，正在处理中..."
        )
        
    except Exception as e:
        logger.error(f"Video caption API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_caption_task(task_id: str):
    """处理视频字幕任务"""
    try:
        task = task_queue[task_id]
        task["status"] = "processing"
        
        request_data = task["request"]
        
        # 处理视频
        results = blip_module.process_video_with_captions(
            video_path=request_data["video_path"],
            fps=request_data["fps"],
            max_frames=request_data["max_frames"],
            export_formats=(["srt"] if request_data["export_srt"] else []) + 
                          (["vtt"] if request_data["export_vtt"] else [])
        )
        
        # 完成任务
        completed_tasks[task_id] = {
            "status": "completed",
            "result": {
                "captions": results["captions"],
                "exported_files": results["exported_files"],
                "count": len(results["captions"])
            }
        }
        
        # 从队列中移除
        del task_queue[task_id]
        
    except Exception as e:
        logger.error(f"Video caption task {task_id} failed: {e}")
        completed_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }
        if task_id in task_queue:
            del task_queue[task_id]


# 提示词助理API
@app.post("/api/prompt-assistant")
async def prompt_assistant_api(request: PromptRequest):
    """提示词助理API"""
    try:
        if not request.base_prompt or len(request.base_prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="请输入基础提示词")
        
        if gpt_module is None:
            raise HTTPException(status_code=500, detail="Nano GPT模块未初始化")
        
        # 执行操作
        if request.operation == "expand":
            results = gpt_module.expand_prompt(
                request.base_prompt,
                style=request.style,
                max_length=request.max_length,
                temperature=request.temperature
            )
        elif request.operation == "optimize":
            results = gpt_module.optimize_prompt(
                request.base_prompt,
                target_style=request.style,
                max_length=request.max_length
            )
        elif request.operation == "negative":
            results = gpt_module.generate_negative_prompt(request.base_prompt)
        else:
            results = [request.base_prompt]
        
        # 分析提示词质量
        analysis = gpt_module.analyze_prompt_quality(request.base_prompt)
        
        return {
            "results": results,
            "analysis": analysis,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Prompt assistant API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 任务状态查询API
@app.get("/api/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """获取任务状态"""
    # 检查队列中的任务
    if task_id in task_queue:
        return TaskStatus(
            task_id=task_id,
            status=task_queue[task_id]["status"]
        )
    
    # 检查已完成的任务
    if task_id in completed_tasks:
        task_data = completed_tasks[task_id]
        return TaskStatus(
            task_id=task_id,
            status=task_data["status"],
            result=task_data.get("result"),
            error=task_data.get("error")
        )
    
    raise HTTPException(status_code=404, detail="任务不存在")


# 获取任务列表
@app.get("/api/tasks")
async def get_tasks():
    """获取任务列表"""
    return {
        "pending": list(task_queue.keys()),
        "completed": list(completed_tasks.keys())
    }


# 下载文件API
@app.get("/api/download/{file_path:path}")
async def download_file(file_path: str):
    """下载生成的文件"""
    full_path = Path(file_path)
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=full_path,
        filename=full_path.name,
        media_type='application/octet-stream'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
