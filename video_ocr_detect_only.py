#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoOCRDetectOnly - 仅检测字幕位置的视频处理类
不进行文字识别，速度更快
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import Optional, Literal

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rapid_ocr_engine import RapidOCREngine
from frame_renderer import FrameRenderer
from utils import BoxInterpolator


class VideoOCRDetectOnly:
    """视频OCR处理类 - 仅检测字幕位置，不识别文字内容"""

    def __init__(
        self,
        # ===== 检测引擎参数 =====
        detection_confidence_threshold: float = 0.3,
        scale_factor: float = 1.0,
        enhance_mode: Optional[Literal["clahe", "binary", "both", "sharpen", "denoise", "denoise_sharpen"]] = "sharpen",
        use_cuda: bool = True,
        det_limit_side_len: int = 960,
        model_type: Literal["MOBILE", "SERVER"] = "MOBILE",

        # ===== 渲染参数 =====
        box_style: str = "red_hollow",

        # ===== 跳帧参数 =====
        skip_frames: int = 0,

        # ===== 插值参数 =====
        interpolate_mode: str = "union",

        # ===== 输出参数 =====
        keep_audio: bool = False,
        verbose: bool = True,
    ):
        """
        初始化 VideoOCRDetectOnly

        Args:
            # 检测引擎参数
            detection_confidence_threshold: 检测置信度阈值(0-1)，推荐0.3-0.5
            scale_factor: 缩放因子(0-1)，1.0=原尺寸，0.5=缩小50%
            enhance_mode: 图像预处理模式
                - None: 不处理
                - "clahe": 对比度增强
                - "binary": 二值化
                - "both": CLAHE + 二值化
                - "sharpen": CLAHE + 锐化（推荐）
                - "denoise": 去噪 + CLAHE
                - "denoise_sharpen": 去噪 + CLAHE + 锐化
            use_cuda: 是否使用GPU加速
            det_limit_side_len: 检测时图像最长边限制
            model_type: 模型类型选择
                - "MOBILE": 移动端模型（速度快，准确率中等，推荐）
                - "SERVER": 服务器模型（速度慢2-3倍，准确率提升5-10%）

            # 渲染参数
            box_style: 标注框样式
                - "red_hollow": 红色空心框
                - "green_fill": 绿色半透明填充

            # 跳帧参数
            skip_frames: 跳帧数，0=逐帧，1=每2帧处理一次

            # 插值参数
            interpolate_mode: 插值模式
                - "union": 前后帧取并集
                - "prev": 使用前一帧
                - "next": 使用下一帧

            # 输出参数
            keep_audio: 是否保留原视频音频（暂未实现）
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        self.keep_audio = keep_audio
        self.skip_frames = skip_frames
        self.detection_confidence_threshold = detection_confidence_threshold

        # 初始化检测引擎（仅检测模式）
        if self.verbose:
            print("\n初始化 RapidOCR 引擎（仅检测模式）...")

        self.ocr_engine = RapidOCREngine(
            detect_only=True,  # 仅检测，不识别
            confidence_threshold=detection_confidence_threshold,  # 检测置信度阈值
            scale_factor=scale_factor,
            enhance_mode=enhance_mode,
            use_det=True,
            use_cls=False,
            use_rec=False,  # 禁用识别
            det_limit_side_len=det_limit_side_len,
            det_limit_type="max",
            use_cuda=use_cuda,
            rec_batch_num=1,  # 不进行识别，设置为1
            model_type=model_type,  # 模型类型选择
        )

        # 初始化渲染器
        self.renderer = FrameRenderer(box_style=box_style)

        # 初始化插值器
        self.interpolator = BoxInterpolator(interpolate_mode=interpolate_mode)

        if self.verbose:
            print(f"检测配置: 检测置信度阈值={detection_confidence_threshold}, 缩放因子={scale_factor}, 增强模式={enhance_mode}, GPU加速={use_cuda}, 模型类型={model_type}")
            print(f"渲染配置: 框样式={box_style}")
            print(f"跳帧配置: skip_frames={skip_frames}")
            print(f"模式: 仅检测字幕位置（不识别文字内容）")

    def process(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        max_frames: Optional[int] = None,
    ):
        """
        处理视频文件

        Args:
            input_path: 输入视频路径
            output_dir: 输出目录，默认为输入视频同目录下的output文件夹
            max_frames: 最大处理帧数，None=处理全部
        """
        if not os.path.exists(input_path):
            print(f"错误: 文件不存在 {input_path}")
            return

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"处理视频: {input_path}")
            print(f"跳帧设置: {self.skip_frames} (每 {self.skip_frames+1} 帧处理一次)")
            print(f"{'='*60}\n")

        # 打开视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {input_path}")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        if self.verbose:
            print(f"视频信息:")
            print(f"  分辨率: {width}x{height}")
            print(f"  帧率: {fps:.2f} FPS")
            print(f"  总帧数: {total_frames}\n")

        # 准备输出目录
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_path), "output")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_annotated = os.path.join(output_dir, f"{base_name}_detect_only_annotated.mp4")
        output_mask = os.path.join(output_dir, f"{base_name}_detect_only_mask.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_annotated = cv2.VideoWriter(output_annotated, fourcc, fps, (width, height))
        out_mask = cv2.VideoWriter(output_mask, fourcc, fps, (width, height))

        # 处理视频
        if self.verbose:
            print(f"\n开始处理视频 (逐帧模式)...")

        start_time = time.time()
        last_log_time = start_time

        frame_id = 0
        buffer = []

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_id >= max_frames):
                break

            # 跳帧逻辑
            should_process = (self.skip_frames == 0) or (frame_id % (self.skip_frames + 1) == 0)

            if should_process:
                # 处理单帧
                boxes, detections = self.ocr_engine.process_frame(frame)

                buffer.append({
                    "frame": frame,
                    "frame_id": frame_id,
                    "boxes": list(boxes),
                    "detections": list(detections)
                })

                if self.verbose:
                    print(f"Frame {frame_id}: 检测到 {len(boxes)} 个文本框")
            else:
                # 跳过的帧直接添加（稍后使用插值）
                buffer.append({
                    "frame": frame,
                    "frame_id": frame_id,
                    "boxes": [],
                    "detections": [],
                })

            # 3 帧滑动窗口插值
            if len(buffer) >= 3:
                prev_data = buffer[0]
                curr_data = buffer[1]
                next_data = buffer[2]

                new_boxes, new_detections, was_interp = self.interpolator.interpolate_frame(
                    prev_data["boxes"], curr_data["boxes"], next_data["boxes"],
                    prev_data["detections"], curr_data["detections"], next_data["detections"]
                )
                if was_interp:
                    curr_data["boxes"] = new_boxes
                    curr_data["detections"] = new_detections

                # 渲染并写入
                annotated_frame, mask_frame = self.renderer.draw_boxes(prev_data["frame"], prev_data["boxes"])
                out_annotated.write(annotated_frame)
                if mask_frame is not None:
                    out_mask.write(mask_frame)
                else:
                    # 如果没有检测到字幕，写入全黑帧
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    out_mask.write(black_frame)
                buffer.pop(0)

            frame_id += 1

            # 进度日志
            if self.verbose:
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    elapsed = current_time - start_time
                    fps_current = frame_id / elapsed if elapsed > 0 else 0
                    percent = frame_id / total_frames * 100
                    eta = (total_frames - frame_id) / fps_current if fps_current > 0 else 0
                    print(f"Progress: {frame_id}/{total_frames} ({percent:.1f}%) | FPS: {fps_current:.2f} | ETA: {eta:.1f}s")
                    last_log_time = current_time

        # 处理剩余缓冲区
        for data in buffer:
            annotated_frame, mask_frame = self.renderer.draw_boxes(data["frame"], data["boxes"])
            out_annotated.write(annotated_frame)
            if mask_frame is not None:
                out_mask.write(mask_frame)
            else:
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                out_mask.write(black_frame)

        # 清理
        cap.release()
        out_annotated.release()
        out_mask.release()

        total_time = time.time() - start_time
        avg_fps = frame_id / total_time if total_time > 0 else 0

        if self.verbose:
            print(f"\n处理完成！")
            print(f"总耗时: {total_time:.2f}s")
            print(f"平均 FPS: {avg_fps:.2f}")
            print(f"输出标注视频: {output_annotated}")
            print(f"输出遮罩视频: {output_mask}")

            # 显示插值统计
            stats = self.interpolator.get_stats()
            print(f"\n插值统计:")
            print(f"  补全帧数: {stats['frames_interpolated']}")
            print(f"  补全框数: {stats['boxes_added']}")


def main():
    """示例：使用 VideoOCRDetectOnly 处理视频"""

    # 创建 VideoOCRDetectOnly 实例
    ocr = VideoOCRDetectOnly(
        # ===== 检测引擎参数 =====
        detection_confidence_threshold=0.3,
        scale_factor=1.0,
        enhance_mode="sharpen",
        use_cuda=True,
        det_limit_side_len=960,

        # ===== 渲染参数 =====
        box_style="red_hollow",

        # ===== 跳帧参数 =====
        skip_frames=0,

        # ===== 输出参数 =====
        verbose=True,
    )

    # 处理视频
    ocr.process(
        input_path="test.mp4",
        max_frames=None,
    )


if __name__ == "__main__":
    main()
