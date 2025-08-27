# DALLE-mini 完整GUI应用
# 包含参数配置和图像生成功能

# By cursor
# TODO @https://github.com/Asp-5a52 解决爆内存问题

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
import threading
from PIL import Image, ImageTk
import tkinter.font as tkfont
from dalle_gerenate_img import Text2imgProcess


class DALLECompleteGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DALLE-mini 图像生成工具")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # 配置文件路径
        self.config_file = "dalle_config.json"
        
        # 默认配置
        self.default_config = {
            "model_path": "./models",
            "device": "auto",
            "is_mega": False,
            "is_reusable": True,
            "dtype": "float32",
            "n_predictions": 4,
            "condition_scale": 10.0,
            "temperature": 1.0,
            "top_k": 256,
            "grid_size": 1,
            "is_seamless": False,
            "output_dir": "./generated_images",
            "auto_save": True,
            "auto_display": True
        }
        
        # 当前配置
        self.current_config = self.load_config()
        
        # 处理器实例
        self.processor = None
        
        # 创建界面
        self.create_widgets()
        self.apply_config()
        
    def create_widgets(self):
        """创建GUI界面元素"""
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="DALLE-mini 图像生成工具", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # 创建左侧控制面板
        self.create_control_panel(main_frame)
        
        # 创建右侧图像显示区域
        self.create_image_panel(main_frame)
        
        # 创建底部日志区域
        self.create_log_panel(main_frame)
        
        # 创建底部按钮区域
        self.create_bottom_buttons(main_frame)
        
    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.LabelFrame(parent, text="参数控制", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        control_frame.columnconfigure(1, weight=1)
        
        # 提示词输入区域
        prompt_frame = ttk.LabelFrame(control_frame, text="图像提示", padding="5")
        prompt_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        prompt_frame.columnconfigure(0, weight=1)
        
        ttk.Label(prompt_frame, text="输入提示词:").grid(row=0, column=0, sticky=tk.W)
        self.prompt_var = tk.StringVar()
        self.prompt_entry = ttk.Entry(prompt_frame, textvariable=self.prompt_var, width=40)
        self.prompt_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(2, 5))
        
        # 预设提示词按钮
        preset_frame = ttk.Frame(prompt_frame)
        preset_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        presets = ["A beautiful sunset over mountains",
            "A cute cat sitting in a garden",
            "A futuristic city with flying cars"]
        for i, preset in enumerate(presets):
            ttk.Button(preset_frame, text=preset, 
                      command=lambda p=preset: self.prompt_var.set(p)).pack(side=tk.LEFT, padx=2)
        
        # 模型设置区域
        model_frame = ttk.LabelFrame(control_frame, text="模型设置", padding="5")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        # 模型路径
        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_path_var = tk.StringVar()
        path_frame = ttk.Frame(model_frame)
        path_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        path_frame.columnconfigure(0, weight=1)
        
        self.model_path_entry = ttk.Entry(path_frame, textvariable=self.model_path_var)
        self.model_path_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(path_frame, text="浏览", command=self.browse_model_path).grid(row=0, column=1)
        
        # 设备选择
        ttk.Label(model_frame, text="计算设备:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.device_var = tk.StringVar()
        device_combo = ttk.Combobox(model_frame, textvariable=self.device_var, 
                                   values=["auto", "cpu", "cuda", "mps"], state="readonly", width=15)
        device_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # 生成参数区域
        gen_frame = ttk.LabelFrame(control_frame, text="生成参数", padding="5")
        gen_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        gen_frame.columnconfigure(1, weight=1)
        
        # 生成数量
        ttk.Label(gen_frame, text="生成数量:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.n_predictions_var = tk.IntVar()
        predictions_spin = ttk.Spinbox(gen_frame, from_=1, to=6, textvariable=self.n_predictions_var, width=10)
        predictions_spin.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # 条件缩放
        ttk.Label(gen_frame, text="条件缩放:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.condition_scale_var = tk.DoubleVar()
        scale_spin = ttk.Spinbox(gen_frame, from_=1.0, to=20.0, increment=0.5, 
                                textvariable=self.condition_scale_var, width=10)
        scale_spin.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # 温度参数
        ttk.Label(gen_frame, text="温度:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.temperature_var = tk.DoubleVar()
        temp_spin = ttk.Spinbox(gen_frame, from_=0.1, to=2.0, increment=0.1, 
                               textvariable=self.temperature_var, width=10)
        temp_spin.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # 输出设置区域
        output_frame = ttk.LabelFrame(control_frame, text="输出设置", padding="5")
        output_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        output_frame.columnconfigure(1, weight=1)
        
        # 输出目录
        ttk.Label(output_frame, text="输出目录:").grid(row=0, column=0, sticky=tk.W, pady=2)
        output_path_frame = ttk.Frame(output_frame)
        output_path_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        output_path_frame.columnconfigure(0, weight=1)
        
        self.output_dir_var = tk.StringVar()
        self.output_dir_entry = ttk.Entry(output_path_frame, textvariable=self.output_dir_var)
        self.output_dir_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(output_path_frame, text="浏览", command=self.browse_output_dir).grid(row=0, column=1)
        
        # 自动保存和显示
        self.auto_save_var = tk.BooleanVar(value=True)
        save_check = ttk.Checkbutton(output_frame, text="自动保存", variable=self.auto_save_var)
        save_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        self.auto_display_var = tk.BooleanVar(value=True)
        display_check = ttk.Checkbutton(output_frame, text="自动显示", variable=self.auto_display_var)
        display_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
    def create_image_panel(self, parent):
        """创建右侧图像显示区域"""
        image_frame = ttk.LabelFrame(parent, text="生成的图像", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)
        
        # 图像标题
        self.image_title_var = tk.StringVar(value="等待生成图像...")
        title_label = ttk.Label(image_frame, textvariable=self.image_title_var, 
                               font=("Arial", 12, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # 图像显示区域
        self.image_canvas = tk.Canvas(image_frame, bg="white", relief=tk.SUNKEN, bd=1)
        self.image_canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        v_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set)
        
        # 图像容器
        self.image_container = ttk.Frame(self.image_canvas)
        self.image_canvas.create_window((0, 0), window=self.image_container, anchor=tk.NW)
        
        # 绑定事件
        self.image_container.bind("<Configure>", lambda e: self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all")))
        
    def create_log_panel(self, parent):
        """创建底部日志区域"""
        log_frame = ttk.LabelFrame(parent, text="运行日志", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 清空日志按钮
        ttk.Button(log_frame, text="清空日志", command=self.clear_log).grid(row=1, column=0, pady=(5, 0))
        
    def create_bottom_buttons(self, parent):
        """创建底部按钮区域"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        # 生成按钮
        generate_btn = ttk.Button(button_frame, text="🎨 生成图像", 
                                 command=self.generate_images, style="Accent.TButton")
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        # 配置按钮
        ttk.Button(button_frame, text="💾 保存配置", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📂 加载配置", command=self.load_config_gui).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 重置默认", command=self.reset_to_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🗑️ 一键清空", command=self.clear_config).pack(side=tk.LEFT, padx=5)
        
    def browse_model_path(self):
        """浏览模型路径"""
        path = filedialog.askdirectory(title="选择模型目录")
        if path:
            self.model_path_var.set(path)
            
    def browse_output_dir(self):
        """浏览输出目录"""
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir_var.set(path)
            
    def log(self, message):
        """添加日志信息"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        
    def apply_config(self):
        """应用配置到界面"""
        config = self.current_config
        
        # 模型设置
        self.model_path_var.set(config.get("model_path", self.default_config["model_path"]))
        self.device_var.set(config.get("device", self.default_config["device"]))
        
        # 生成参数
        self.n_predictions_var.set(config.get("n_predictions", self.default_config["n_predictions"]))
        self.condition_scale_var.set(config.get("condition_scale", self.default_config["condition_scale"]))
        self.temperature_var.set(config.get("temperature", self.default_config["temperature"]))
        
        # 输出设置
        self.output_dir_var.set(config.get("output_dir", self.default_config["output_dir"]))
        self.auto_save_var.set(config.get("auto_save", self.default_config["auto_save"]))
        self.auto_display_var.set(config.get("auto_display", self.default_config["auto_display"]))
        
    def get_current_config(self):
        """获取当前界面配置"""
        return {
            "model_path": self.model_path_var.get(),
            "device": self.device_var.get(),
            "n_predictions": self.n_predictions_var.get(),
            "condition_scale": self.condition_scale_var.get(),
            "temperature": self.temperature_var.get(),
            "output_dir": self.output_dir_var.get(),
            "auto_save": self.auto_save_var.get(),
            "auto_display": self.auto_display_var.get()
        }
        
    def save_config(self):
        """保存配置"""
        try:
            config = self.get_current_config()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            self.current_config = config
            self.log(f"✅ 配置已保存到 {self.config_file}")
            messagebox.showinfo("成功", f"配置已保存到 {self.config_file}")
            
        except Exception as e:
            self.log(f"❌ 保存失败: {str(e)}")
            messagebox.showerror("错误", f"保存配置失败: {str(e)}")
            
    def load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 合并默认配置
                for key, value in self.default_config.items():
                    if key not in config:
                        config[key] = value
                        
                return config
            else:
                return self.default_config.copy()
                
        except Exception as e:
            messagebox.showwarning("警告", f"加载配置失败: {str(e)}，使用默认配置")
            return self.default_config.copy()
            
    def load_config_gui(self):
        """GUI加载配置"""
        try:
            self.current_config = self.load_config()
            self.apply_config()
            self.log("✅ 配置已加载")
            messagebox.showinfo("成功", "配置已加载")
        except Exception as e:
            self.log(f"❌ 加载失败: {str(e)}")
            messagebox.showerror("错误", f"加载配置失败: {str(e)}")
            
    def reset_to_default(self):
        """重置为默认配置"""
        if messagebox.askyesno("确认", "确定要重置为默认配置吗？"):
            self.current_config = self.default_config.copy()
            self.apply_config()
            self.log("🔄 已重置为默认配置")
            
    def clear_config(self):
        """一键清空配置"""
        if messagebox.askyesno("确认", "确定要清空所有配置吗？"):
            self.prompt_var.set("")
            self.model_path_var.set("")
            self.output_dir_var.set("")
            self.n_predictions_var.set(0)
            self.condition_scale_var.set(0.0)
            self.temperature_var.set(0.0)
            self.device_var.set("")
            self.auto_save_var.set(False)
            self.auto_display_var.set(False)
            self.log("🗑️ 配置已清空")
            
    def generate_images(self):
        """生成图像"""
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("警告", "请输入图像提示词")
            return
            
        # 在新线程中运行生成任务
        thread = threading.Thread(target=self._generate_images_thread, args=(prompt,))
        thread.daemon = True
        thread.start()
        
    def _generate_images_thread(self, prompt):
        """在后台线程中生成图像"""
        try:
            self.log("🚀 开始生成图像...")
            self.log(f"📝 提示词: {prompt}")
            
            # 获取配置
            config = self.get_current_config()
            self.log(f"⚙️ 生成参数: 数量={config['n_predictions']}, 条件缩放={config['condition_scale']}")
            
            # 初始化处理器
            if self.processor is None:
                self.log("📦 初始化模型...")
                self.processor = Text2imgProcess(model_path=config['model_path'])
                self.log("✅ 模型初始化完成")
            
            # 生成图像
            images = self.processor.generate_image(
                prompt=prompt,
                n_predictions=config['n_predictions'],
                condition_scale=config['condition_scale']
            )
            
            if images:
                self.log(f"🎉 成功生成 {len(images)} 张图像")
                
                # 显示图像
                if config['auto_display']:
                    self.display_images(images, prompt)
                    
                # 保存图像
                if config['auto_save']:
                    self.processor.save_images(images, prompt, config['output_dir'])
                    self.log(f"💾 图像已保存到 {config['output_dir']}")
            else:
                self.log("❌ 图像生成失败")
                
        except Exception as e:
            self.log(f"❌ 生成过程中出错: {str(e)}")
            messagebox.showerror("错误", f"图像生成失败: {str(e)}")
            
    def display_images(self, images, prompt):
        """在界面中显示图像"""
        try:
            # 清空之前的图像
            for widget in self.image_container.winfo_children():
                widget.destroy()
                
            # 更新标题
            self.image_title_var.set(f"生成的图像 - {prompt}")
            
            # 显示图像
            for i, image in enumerate(images):
                # 调整图像大小
                display_size = (200, 200)
                resized_image = image.resize(display_size, Image.Resampling.LANCZOS)
                
                # 转换为PhotoImage
                photo = ImageTk.PhotoImage(resized_image)
                
                # 创建图像标签
                img_frame = ttk.Frame(self.image_container)
                img_frame.grid(row=0, column=i, padx=5, pady=5)
                
                img_label = ttk.Label(img_frame, text=f"图像 {i+1}", font=("Arial", 10))
                img_label.pack()
                
                img_label = ttk.Label(img_frame, image=photo)
                img_label.image = photo  # 保持引用
                img_label.pack()
                
            # 调整滚动区域
            self.image_container.update_idletasks()
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            self.log(f"❌ 显示图像时出错: {str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = DALLECompleteGUI(root)
    
    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    main()
