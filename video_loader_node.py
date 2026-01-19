#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Video Loader Nodes - 使用 OpenCV 加载视频并支持预览
"""

import os
import numpy as np
import torch
import cv2
import folder_paths
from pathlib import Path


class VideoLoader:
    """
    使用 OpenCV 加载视频文件的 ComfyUI 节点
    支持从 input 文件夹选择视频文件
    """

    def __init__(self):
        self.video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v']

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 input 目录中的视频文件
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        # 过滤出视频文件
        files = folder_paths.filter_files_content_types(files, ["video"])

        return {
            "required": {
                # 视频文件选择（支持上传）
                "video": (sorted(files), {"video_upload": True}),

                # 加载选项
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 999999}),  # 0 = 全部
                "skip_frames": ("INT", {"default": 0, "min": 0, "max": 100}),  # 跳帧加载

                # 缩放选项
                "resize_mode": (["none", "scale", "fit"], {"default": "none"}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("images", "frame_count", "width", "fps", "total_frames")
    FUNCTION = "load_video"
    CATEGORY = "SubtitleDetector"

    def get_video_info(self, video_path):
        """使用 OpenCV 获取视频信息"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()
        return width, height, fps, total_frames

    def resize_frame(self, frame, resize_mode, target_width, target_height):
        """调整帧大小"""
        if resize_mode == "none":
            return frame

        h, w = frame.shape[:2]

        if resize_mode == "scale":
            # 直接缩放到目标尺寸
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            return resized

        elif resize_mode == "fit":
            # 等比缩放，保持宽高比
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # 创建黑色背景
            result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            # 居中放置
            y_offset = (target_height - new_h) // 2
            x_offset = (target_width - new_w) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return result

        return frame

    def load_video(self, video, start_frame=0, max_frames=0, skip_frames=0,
                   resize_mode="none", target_width=512, target_height=512):
        """
        加载视频文件

        Args:
            video: 视频文件名
            start_frame: 起始帧
            max_frames: 最大加载帧数（0=全部）
            skip_frames: 跳帧数（0=逐帧，1=每2帧加载1帧）
            resize_mode: 缩放模式
            target_width: 目标宽度
            target_height: 目标高度
        """
        # 获取视频文件完整路径
        video_path = folder_paths.get_annotated_filepath(video)

        # 验证文件存在
        if not os.path.exists(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")

        file_ext = Path(video_path).suffix.lower()
        if file_ext not in self.video_extensions:
            raise ValueError(f"不支持的视频格式: {file_ext}")

        print(f"\n{'='*60}")
        print(f"视频加载器 (OpenCV)")
        print(f"{'='*60}")
        print(f"文件: {video_path}")

        # 获取视频信息
        orig_width, orig_height, fps, total_frames = self.get_video_info(video_path)

        print(f"视频信息:")
        print(f"  分辨率: {orig_width}x{orig_height}")
        print(f"  帧率: {fps:.2f} FPS")
        print(f"  总帧数: {total_frames}")
        print(f"加载设置:")
        print(f"  起始帧: {start_frame}")
        print(f"  最大帧数: {max_frames if max_frames > 0 else '全部'}")
        print(f"  跳帧: {skip_frames} (每 {skip_frames+1} 帧加载一次)")
        print(f"  缩放模式: {resize_mode}")
        if resize_mode != "none":
            print(f"  目标尺寸: {target_width}x{target_height}")

        # 使用 OpenCV 加载视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 跳到起始帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frame_id = start_frame
        loaded_count = 0

        print(f"\n开始加载视频帧...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检查是否达到最大帧数
            if max_frames > 0 and loaded_count >= max_frames:
                break

            # 跳帧逻辑
            should_load = (skip_frames == 0) or ((frame_id - start_frame) % (skip_frames + 1) == 0)

            if should_load:
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 调整大小
                frame_resized = self.resize_frame(frame_rgb, resize_mode, target_width, target_height)

                # 转换为 float32 [0, 1]
                frame_tensor = frame_resized.astype(np.float32) / 255.0

                frames.append(frame_tensor)
                loaded_count += 1

                # 进度显示
                if loaded_count % 100 == 0 or loaded_count == 1:
                    print(f"  已加载 {loaded_count} 帧...")

            frame_id += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError("未能加载任何帧，请检查起始帧位置或视频文件")

        # 转换为 torch tensor [B, H, W, C]
        frames_array = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array)

        final_height, final_width = frames_tensor.shape[1:3]

        print(f"\n✅ 加载完成！")
        print(f"  加载帧数: {loaded_count}")
        print(f"  输出尺寸: {final_width}x{final_height}")
        print(f"{'='*60}\n")

        return (frames_tensor, loaded_count, final_width, fps, total_frames)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        """检查视频文件是否改变"""
        video_path = folder_paths.get_annotated_filepath(video)
        if os.path.exists(video_path):
            return os.path.getmtime(video_path)
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        """验证输入"""
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


class VideoCombine:
    """
    将图片序列组合为视频并预览
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "combine"
    CATEGORY = "SubtitleDetector"

    def combine(self, images, frame_rate):
        """
        将图片序列组合为视频并生成预览

        Args:
            images: 图片张量 [B, H, W, C]
            frame_rate: 帧率
        """
        import time

        print(f"\n{'='*60}")
        print(f"视频预览")
        print(f"{'='*60}")
        print(f"帧数: {len(images)}")
        print(f"帧率: {frame_rate} FPS")
        print(f"分辨率: {images.shape[2]}x{images.shape[1]} (宽x高)")

        # 创建临时预览文件
        timestamp = int(time.time() * 1000)
        temp_dir = folder_paths.get_temp_directory()
        preview_file = f"preview_{timestamp}.mp4"
        preview_path = os.path.join(temp_dir, preview_file)

        # 转换图片为 numpy 数组 uint8
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)

        # 获取尺寸
        height, width = images_np.shape[1:3]

        # 尝试使用不同的编码器,按优先级排序
        codecs_to_try = [
            ('avc1', 'H.264'),  # H.264 编码器 (最兼容)
            ('mp4v', 'MPEG-4'),  # MPEG-4 编码器
            ('XVID', 'Xvid'),    # Xvid 编码器
        ]

        out = None
        used_codec = None

        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_out = cv2.VideoWriter(preview_path, fourcc, frame_rate, (width, height))
                if test_out.isOpened():
                    out = test_out
                    used_codec = codec_desc
                    print(f"使用编码器: {codec_desc}")
                    break
                else:
                    test_out.release()
            except Exception as e:
                print(f"编码器 {codec_desc} 不可用: {e}")
                continue

        if out is None or not out.isOpened():
            raise ValueError(f"无法创建预览文件 (所有编码器都失败): {preview_path}")

        # 写入帧
        for frame in images_np:
            # RGB -> BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

        print(f"✅ 预览生成完成！")
        print(f"{'='*60}\n")

        # 返回预览信息（使用 gifs 格式，兼容 ComfyUI 的视频预览）
        preview = {
            "filename": preview_file,
            "subfolder": "",
            "type": "temp",
            "format": "video/mp4",
        }
        return {"ui": {"gifs": [preview]}}


class SaveVideo:
    """
    保存视频到文件
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI"}),
                "format": (["mp4", "webm", "avi", "mkv"], {"default": "mp4"}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "SubtitleDetector"

    def save(self, images, frame_rate, filename_prefix, format):
        """
        保存视频到文件

        Args:
            images: 图片张量 [B, H, W, C]
            frame_rate: 帧率
            filename_prefix: 文件名前缀
            format: 视频格式
        """

        # 获取输出路径
        output_dir = folder_paths.get_output_directory()

        # 处理文件名前缀
        if "/" in filename_prefix or "\\" in filename_prefix:
            # 包含子目录
            parts = filename_prefix.replace("\\", "/").split("/")
            subfolder = "/".join(parts[:-1])
            filename = parts[-1]
            full_output_folder = os.path.join(output_dir, subfolder)
            os.makedirs(full_output_folder, exist_ok=True)
        else:
            full_output_folder = output_dir
            filename = filename_prefix
            subfolder = ""

        # 查找下一个可用的计数器
        counter = 1
        while True:
            file_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.{format}")
            if not os.path.exists(file_path):
                break
            counter += 1

        file = f"{filename}_{counter:05}_.{format}"
        output_path = os.path.join(full_output_folder, file)

        print(f"\n{'='*60}")
        print(f"保存视频")
        print(f"{'='*60}")
        print(f"输出: {output_path}")
        print(f"帧数: {len(images)}")
        print(f"帧率: {frame_rate} FPS")
        print(f"格式: {format}")

        # 转换图片为 numpy 数组 uint8
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)

        # 获取尺寸
        height, width = images_np.shape[1:3]
        print(f"分辨率: {width}x{height}")

        # 设置编解码器
        codec_map = {
            "mp4": "mp4v",
            "webm": "VP80",
            "avi": "XVID",
            "mkv": "XVID",
        }
        fourcc = cv2.VideoWriter_fourcc(*codec_map.get(format, "mp4v"))

        # 创建视频写入器
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        if not out.isOpened():
            raise ValueError(f"无法创建视频文件: {output_path}")

        # 写入帧
        for i, frame in enumerate(images_np):
            # RGB -> BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  已写入 {i+1}/{len(images)} 帧...")

        out.release()

        print(f"\n✅ 视频保存完成！")
        print(f"{'='*60}\n")

        # 不返回预览,让用户自己决定是否预览
        return {}


class VideoPreviewHelper:
    """
    视频预览辅助节点
    从视频张量中提取特定帧用于预览
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_image",)
    FUNCTION = "get_frame"
    CATEGORY = "SubtitleDetector"

    def get_frame(self, images, frame_index):
        """提取指定帧用于预览"""
        total_frames = len(images)

        # 确保索引有效
        if frame_index >= total_frames:
            frame_index = total_frames - 1
        if frame_index < 0:
            frame_index = 0

        # 提取单帧并添加批次维度
        frame = images[frame_index:frame_index+1]

        print(f"预览帧 {frame_index}/{total_frames-1}")

        return (frame,)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "VideoLoader": VideoLoader,
    "VideoCombine": VideoCombine,
    "SaveVideo": SaveVideo,
    "VideoPreviewHelper": VideoPreviewHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLoader": "📹 Video Loader",
    "VideoCombine": "🎬 Video Combine",
    "SaveVideo": "💾 Save Video",
    "VideoPreviewHelper": "🔍 Video Preview Helper",
}
