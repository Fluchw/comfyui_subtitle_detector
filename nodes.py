#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Subtitle Detector Node - 使用 RapidOCR 进行字幕检测
重写版：使用统一的 VideoOCR API
"""

import time
import torch
import numpy as np
import cv2

from .video_ocr import OCRMode, ModelType, EnhanceMode, BoxStyle
from .rapid_ocr_engine import RapidOCREngine
from .frame_renderer import FrameRenderer
from .utils import BoxInterpolator

# ComfyUI 进度条支持
try:
    from comfy.utils import ProgressBar
    HAS_PROGRESS_BAR = True
except ImportError:
    HAS_PROGRESS_BAR = False


class SubtitleDetectorRapidOCR:
    """
    ComfyUI 字幕检测节点 - 使用 RapidOCR

    推荐默认配置：
    - OCR模式: DETECT_ONLY (仅检测，最快)
    - 模型类型: MOBILE (速度快)
    - 增强模式: SHARPEN (锐化)
    - 置信度阈值: 0.3 (检测模式推荐)
    - 缩放因子: 0.8 (降低分辨率提升速度)
    - 标注样式: RED_HOLLOW (红色空心框)
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 输入视频流
                "images": ("IMAGE",),

                # ===== OCR 模式 =====
                "ocr_mode": (["detect_only", "detect_rec", "full"], {"default": "detect_only"}),

                # ===== 模型类型 =====
                "model_type": (["MOBILE", "SERVER"], {"default": "MOBILE"}),

                # ===== 置信度阈值 =====
                "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),

                # ===== 缩放因子 =====
                "scale_factor": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),

                # ===== 图像增强 =====
                "enhance_mode": (["sharpen", "clahe", "binary", "both", "denoise", "denoise_sharpen", "None"], {"default": "sharpen"}),

                # ===== 标注样式 =====
                "box_style": (["red_hollow", "green_fill", "mask"], {"default": "red_hollow"}),

                # ===== 跳帧设置 =====
                "skip_frames": ("INT", {"default": 0, "min": 0, "max": 10}),

                # ===== GPU 加速 =====
                "use_cuda": ("BOOLEAN", {"default": True}),

                # ===== 插值模式 =====
                "interpolate_mode": (["union", "linear"], {"default": "union"}),
            },
            "optional": {
                # 识别批处理大小 (仅在识别模式下生效)
                "rec_batch_num": ("INT", {"default": 6, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("annotated_images", "subtitle_mask")
    FUNCTION = "process_images"
    CATEGORY = "SubtitleDetector"

    def process_images(self, images, ocr_mode, model_type, confidence_threshold, scale_factor,
                      enhance_mode, box_style, skip_frames, use_cuda, interpolate_mode,
                      rec_batch_num=6):


        # 转换枚举值
        enhance_mode_val = None if enhance_mode == "None" else enhance_mode

        print(f"SubtitleDetector (RapidOCR) - 配置:")
        print(f"  OCR模式: {ocr_mode}")
        print(f"  模型类型: {model_type}")
        print(f"  置信度阈值: {confidence_threshold}")
        print(f"  缩放因子: {scale_factor}")
        print(f"  增强模式: {enhance_mode_val}")
        print(f"  标注样式: {box_style}")
        print(f"  跳帧: {skip_frames}")
        print(f"  GPU加速: {use_cuda}")

        # 根据 OCR 模式设置参数
        if ocr_mode == "detect_only":
            use_det, use_cls, use_rec = True, False, False
            detect_only = True
        elif ocr_mode == "detect_rec":
            use_det, use_cls, use_rec = True, False, True
            detect_only = False
        elif ocr_mode == "full":
            use_det, use_cls, use_rec = True, True, True
            detect_only = False
        else:
            use_det, use_cls, use_rec = True, False, False
            detect_only = True

        # 初始化 RapidOCR 引擎
        ocr_engine = RapidOCREngine(
            detect_only=detect_only,
            confidence_threshold=confidence_threshold,
            scale_factor=scale_factor,
            enhance_mode=enhance_mode_val,
            use_det=use_det,
            use_cls=use_cls,
            use_rec=use_rec,
            det_limit_side_len=960,
            det_limit_type="max",
            use_cuda=use_cuda,
            rec_batch_num=rec_batch_num,
            model_type=model_type
        )

        # 初始化渲染器
        renderer = FrameRenderer(box_style=box_style)

        # 初始化插值器
        interpolator = BoxInterpolator(interpolate_mode=interpolate_mode)

        # ComfyUI 输入格式: Tensor [Batch, H, W, C], range 0-1
        batch_size = len(images)
        h, w = images.shape[1], images.shape[2]
        input_device = images.device

        # 流式处理：预分配输出 tensor（在 CPU 上避免内存溢出）
        print(f"\nSubtitleDetector: 流式处理 {batch_size} 帧...")
        output_images_tensor = torch.zeros((batch_size, h, w, 3), dtype=torch.float32)
        output_masks_tensor = torch.zeros((batch_size, h, w), dtype=torch.float32)

        # 辅助函数
        def tensor_to_cv2(img_tensor):
            """RGB float -> BGR uint8"""
            img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        def cv2_to_numpy(img_cv2):
            """BGR uint8 -> RGB float numpy"""
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            return img_rgb.astype(np.float32) / 255.0

        # 开始处理
        start_time = time.time()
        last_log_time = start_time

        # 初始化进度条
        pbar = ProgressBar(batch_size) if HAS_PROGRESS_BAR else None

        buffer = []
        frame_id = 0
        output_idx = 0
        last_boxes = []

        for i in range(batch_size):
            # 逐帧取出并立即转换，减少内存占用
            curr_tensor = images[i]
            frame = tensor_to_cv2(curr_tensor)

            # 跳帧逻辑
            should_process = (skip_frames == 0) or (frame_id % (skip_frames + 1) == 0)

            if should_process:
                boxes, detections = ocr_engine.process_frame(frame)
                last_boxes = list(boxes)
                buffer.append({
                    "frame": frame,
                    "boxes": list(boxes),
                    "detections": list(detections),
                    "skipped": False
                })
            else:
                # 跳过的帧使用上一帧的结果
                buffer.append({
                    "frame": frame,
                    "boxes": list(last_boxes),
                    "detections": [],
                    "skipped": True
                })

            # 3 帧滑动窗口插值
            if len(buffer) >= 3:
                prev_data = buffer[0]
                curr_data = buffer[1]
                next_data = buffer[2]

                is_skipped_frame = curr_data.get("skipped", False)

                if skip_frames > 0 and is_skipped_frame:
                    if prev_data["boxes"] and next_data["boxes"]:
                        new_boxes, new_detections, was_interp = interpolator.interpolate_frame(
                            prev_data["boxes"], curr_data["boxes"], next_data["boxes"],
                            prev_data["detections"], curr_data["detections"], next_data["detections"]
                        )
                        curr_data["boxes"] = new_boxes
                        curr_data["detections"] = new_detections
                else:
                    new_boxes, new_detections, was_interp = interpolator.interpolate_frame(
                        prev_data["boxes"], curr_data["boxes"], next_data["boxes"],
                        prev_data["detections"], curr_data["detections"], next_data["detections"]
                    )
                    if was_interp:
                        curr_data["boxes"] = new_boxes
                        curr_data["detections"] = new_detections

                # 渲染并直接写入预分配的 tensor
                annotated_frame, mask_frame = renderer.draw_boxes(prev_data["frame"], prev_data["boxes"])
                output_images_tensor[output_idx] = torch.from_numpy(cv2_to_numpy(annotated_frame))

                if mask_frame is not None:
                    if len(mask_frame.shape) == 3:
                        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        mask_gray = mask_frame
                    output_masks_tensor[output_idx] = torch.from_numpy(mask_gray.astype(np.float32) / 255.0)

                output_idx += 1
                buffer.pop(0)

            frame_id += 1

            # 更新进度条
            if pbar is not None:
                pbar.update(1)

            # 每秒输出一次进度
            current_time = time.time()
            if current_time - last_log_time >= 1.0 or i == batch_size - 1:
                elapsed = current_time - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                percent = (i + 1) / batch_size * 100
                eta = (batch_size - i - 1) / fps if fps > 0 else 0
                print(f"进度: {i+1}/{batch_size} ({percent:.1f}%) | FPS: {fps:.2f} | 预计剩余: {eta:.1f}s")
                last_log_time = current_time

        # 清空缓冲区
        for data in buffer:
            annotated_frame, mask_frame = renderer.draw_boxes(data["frame"], data["boxes"])
            output_images_tensor[output_idx] = torch.from_numpy(cv2_to_numpy(annotated_frame))

            if mask_frame is not None:
                if len(mask_frame.shape) == 3:
                    mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                else:
                    mask_gray = mask_frame
                output_masks_tensor[output_idx] = torch.from_numpy(mask_gray.astype(np.float32) / 255.0)

            output_idx += 1

        # 如果原输入在 GPU 上，将结果移回 GPU
        if input_device.type != 'cpu':
            output_images_tensor = output_images_tensor.to(input_device)
            output_masks_tensor = output_masks_tensor.to(input_device)

        total_time = time.time() - start_time
        avg_fps = batch_size / total_time if total_time > 0 else 0

        print(f"\n✅ 处理完成！")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  平均 FPS: {avg_fps:.2f}")

        return (output_images_tensor, output_masks_tensor)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "SubtitleDetectorRapidOCR": SubtitleDetectorRapidOCR
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleDetectorRapidOCR": "🎥 Subtitle Detector (RapidOCR v3)"
}
