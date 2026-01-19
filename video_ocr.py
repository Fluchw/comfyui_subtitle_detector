#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoOCR - 视频字幕检测处理类
统一封装所有配置参数，支持多种OCR模式
"""

import os
import time
import cv2
import numpy as np
from typing import Optional, Union
from enum import Enum

# 使用相对导入避免与 ComfyUI 的 utils 模块冲突
from .rapid_ocr_engine import RapidOCREngine
from .frame_renderer import FrameRenderer
from .utils import BoxInterpolator


class OCRMode(Enum):
    """OCR模式枚举（参考RapidOCR官方文档）

    参考: https://github.com/RapidAI/RapidOCRDocs/blob/main/docs/install_usage/rapidocr/usage.md
    """
    DETECT_ONLY = "detect_only"              # 仅检测文字位置（最快，推荐用于生成遮罩）
    DETECT_REC = "detect_rec"                # 检测+识别（跳过方向分类，推荐，默认）
    FULL = "full"                            # 完整OCR（检测+方向分类+识别，最准确但最慢）
    REC_ONLY = "rec_only"                    # 仅识别（需要预先提供文本框位置）
    CUSTOM = "custom"                        # 自定义模式（手动指定use_det/use_cls/use_rec）


class ModelType(Enum):
    """模型类型枚举"""
    MOBILE = "MOBILE"                        # 移动端模型（速度快，准确率中等）
    SERVER = "SERVER"                        # 服务器模型（速度慢2-3倍，准确率提升5-10%）


class EnhanceMode(Enum):
    """图像增强模式枚举"""
    NONE = None                              # 不处理
    CLAHE = "clahe"                          # 对比度增强
    BINARY = "binary"                        # 二值化
    BOTH = "both"                            # CLAHE + 二值化
    SHARPEN = "sharpen"                      # CLAHE + 锐化（推荐）
    DENOISE = "denoise"                      # 去噪 + CLAHE
    DENOISE_SHARPEN = "denoise_sharpen"      # 去噪 + CLAHE + 锐化


class BoxStyle(Enum):
    """标注框样式枚举"""
    RED_HOLLOW = "red_hollow"                # 红色空心框
    GREEN_FILL = "green_fill"                # 绿色半透明填充
    MASK = "mask"                            # 纯黑白遮罩


class VideoOCR:
    """视频OCR处理类 - 集成字幕检测、识别和渲染"""

    def __init__(
        self,
        # ===== OCR 模式参数 =====
        ocr_mode: Union[OCRMode, str] = OCRMode.DETECT_ONLY,  # OCR模式
        use_det: Optional[bool] = None,  # 自定义：是否启用检测
        use_cls: Optional[bool] = None,  # 自定义：是否启用方向分类
        use_rec: Optional[bool] = None,  # 自定义：是否启用识别

        # ===== OCR 引擎参数 =====
        confidence_threshold: float = 0.3,  # 降低默认阈值
        scale_factor: float = 1.0,
        enhance_mode: Union[EnhanceMode, str, None] = EnhanceMode.SHARPEN,  # 图像增强模式
        use_cuda: bool = True,
        rec_batch_num: int = 6,  # 识别批处理大小
        model_type: Union[ModelType, str] = ModelType.MOBILE,  # 模型类型

        # ===== 渲染参数 =====
        box_style: Union[BoxStyle, str] = BoxStyle.RED_HOLLOW,  # 标注框样式

        # ===== 批处理参数 =====
        batch_size: int = 8,
        skip_frames: int = 0,

        # ===== 插值参数 =====
        interpolate_mode: str = "union",

        # ===== 输出参数 =====
        keep_audio: bool = False,
        verbose: bool = True,
    ):
        """
        初始化 VideoOCR

        Args:
            # OCR 模式参数
            ocr_mode: OCR模式，可选值：
                - OCRMode.DETECT_ONLY 或 "detect_only": 仅检测（最快，推荐用于生成遮罩）
                - OCRMode.DETECT_REC 或 "detect_rec": 检测+识别（推荐，默认）
                - OCRMode.FULL 或 "full": 完整OCR（检测+方向分类+识别）
                - OCRMode.REC_ONLY 或 "rec_only": 仅识别
                - OCRMode.CUSTOM 或 "custom": 自定义（需设置use_det/use_cls/use_rec）
            use_det: 自定义模式：是否启用检测（仅在ocr_mode="custom"时使用）
            use_cls: 自定义模式：是否启用方向分类（仅在ocr_mode="custom"时使用）
            use_rec: 自定义模式：是否启用识别（仅在ocr_mode="custom"时使用）

            # OCR 引擎参数
            confidence_threshold: 置信度阈值(0-1)，推荐0.3-0.5
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
            rec_batch_num: 识别批处理大小
            model_type: 模型类型（"MOBILE" 或 "SERVER"）

            # 渲染参数
            box_style: 标注框样式
                - "red_hollow": 红色空心框
                - "green_fill": 绿色半透明填充

            # 批处理参数
            batch_size: 批量处理帧数，GPU推荐4-8
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
        self.batch_size = batch_size
        self.skip_frames = skip_frames

        # 解析枚举参数（支持字符串和枚举）
        if isinstance(ocr_mode, str):
            ocr_mode = OCRMode(ocr_mode)

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        if isinstance(enhance_mode, str):
            enhance_mode = EnhanceMode(enhance_mode)
        elif enhance_mode is None:
            enhance_mode = EnhanceMode.NONE

        if isinstance(box_style, str):
            box_style = BoxStyle(box_style)

        # 根据OCR模式设置use_det/use_cls/use_rec
        if ocr_mode == OCRMode.DETECT_ONLY:
            final_use_det, final_use_cls, final_use_rec = True, False, False
        elif ocr_mode == OCRMode.DETECT_REC:
            final_use_det, final_use_cls, final_use_rec = True, False, True
        elif ocr_mode == OCRMode.FULL:
            final_use_det, final_use_cls, final_use_rec = True, True, True
        elif ocr_mode == OCRMode.REC_ONLY:
            final_use_det, final_use_cls, final_use_rec = False, False, True
        elif ocr_mode == OCRMode.CUSTOM:
            # 自定义模式：使用用户提供的参数
            if use_det is None or use_cls is None or use_rec is None:
                raise ValueError("CUSTOM模式需要明确指定use_det, use_cls, use_rec参数")
            final_use_det, final_use_cls, final_use_rec = use_det, use_cls, use_rec
        else:
            raise ValueError(f"未知的OCR模式: {ocr_mode}")

        self.ocr_mode = ocr_mode
        self.use_det = final_use_det
        self.use_cls = final_use_cls
        self.use_rec = final_use_rec

        # 初始化 OCR 引擎
        if self.verbose:
            mode_desc = {
                OCRMode.DETECT_ONLY: "仅检测模式",
                OCRMode.DETECT_REC: "检测+识别模式",
                OCRMode.FULL: "完整OCR模式（含方向分类）",
                OCRMode.REC_ONLY: "仅识别模式",
                OCRMode.CUSTOM: f"自定义模式 (det={final_use_det}, cls={final_use_cls}, rec={final_use_rec})"
            }
            print(f"\n初始化 RapidOCR 引擎 [{mode_desc[ocr_mode]}]...")

        # 判断是否为仅检测模式
        detect_only = (ocr_mode == OCRMode.DETECT_ONLY)

        self.ocr_engine = RapidOCREngine(
            detect_only=detect_only,
            confidence_threshold=confidence_threshold,
            scale_factor=scale_factor,
            enhance_mode=enhance_mode.value,  # 传递枚举值
            use_det=final_use_det,
            use_cls=final_use_cls,
            use_rec=final_use_rec,
            det_limit_side_len=960,
            det_limit_type="max",
            use_cuda=use_cuda,
            rec_batch_num=rec_batch_num,
            model_type=model_type.value  # 传递枚举值
        )

        # 初始化渲染器
        self.renderer = FrameRenderer(box_style=box_style.value)  # 传递枚举值

        # 初始化插值器
        self.interpolator = BoxInterpolator(interpolate_mode=interpolate_mode)

        if self.verbose:
            print(f"OCR 配置: 模式={ocr_mode.value}, 置信度阈值={confidence_threshold}, 增强模式={enhance_mode.value}, GPU加速={use_cuda}, 模型类型={model_type.value}")
            if not detect_only:
                print(f"识别配置: 批处理={rec_batch_num}")
            print(f"渲染配置: 框样式={box_style.value}")
            print(f"批处理配置: batch_size={batch_size}, skip_frames={skip_frames}")

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
        output_annotated = os.path.join(output_dir, f"{base_name}_annotated.mp4")
        output_mask = os.path.join(output_dir, f"{base_name}_mask.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_annotated = cv2.VideoWriter(output_annotated, fourcc, fps, (width, height))
        out_mask = cv2.VideoWriter(output_mask, fourcc, fps, (width, height))

        # 处理视频
        if self.verbose:
            print(f"\n开始处理视频 (批处理模式, batch_size={self.batch_size})...")

        start_time = time.time()
        last_log_time = start_time

        frame_id = 0
        buffer = []
        frame_batch = []
        frame_data_batch = []

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_id >= max_frames):
                # 处理剩余的批次
                if frame_batch:
                    results = self.ocr_engine.process_frames_batch(frame_batch)
                    for idx, (boxes, detections) in enumerate(results):
                        data = frame_data_batch[idx]
                        data['boxes'] = list(boxes)
                        data['detections'] = list(detections)
                        buffer.append(data)

                        if self.verbose:
                            print(f"Frame {data['frame_id']}: 检测到 {len(boxes)} 个文本框")
                            if len(detections) > 0:
                                for det in detections[:3]:
                                    print(f"  - 文本: '{det['text']}' (置信度: {det['confidence']:.2f})")
                break

            # 跳帧逻辑
            should_process = (self.skip_frames == 0) or (frame_id % (self.skip_frames + 1) == 0)

            if should_process:
                # 添加到批次
                frame_batch.append(frame)
                frame_data_batch.append({
                    "frame": frame,
                    "frame_id": frame_id,
                    "boxes": [],
                    "detections": []
                })

                # 当批次满了，批量处理
                if len(frame_batch) >= self.batch_size:
                    results = self.ocr_engine.process_frames_batch(frame_batch)
                    for idx, (boxes, detections) in enumerate(results):
                        data = frame_data_batch[idx]
                        data['boxes'] = list(boxes)
                        data['detections'] = list(detections)
                        buffer.append(data)

                        if self.verbose:
                            print(f"Frame {data['frame_id']}: 检测到 {len(boxes)} 个文本框")
                            if len(detections) > 0:
                                for det in detections[:3]:
                                    print(f"  - 文本: '{det['text']}' (置信度: {det['confidence']:.2f})")

                    # 清空批次
                    frame_batch = []
                    frame_data_batch = []
            else:
                # 跳过的帧直接添加（稍后使用插值补全）
                buffer.append({
                    "frame": frame,
                    "frame_id": frame_id,
                    "boxes": [],
                    "detections": [],
                    "skipped": True,  # 标记为跳帧
                })

            # 3 帧滑动窗口插值
            # 如果启用了跳帧(skip_frames > 0)，则自动为跳过的帧补全检测框
            if len(buffer) >= 3:
                prev_data = buffer[0]
                curr_data = buffer[1]
                next_data = buffer[2]

                # 检查中间帧是否需要插值
                is_skipped_frame = curr_data.get("skipped", False)

                # 如果启用了跳帧，优先为跳过的帧补全框
                # 如果未启用跳帧，插值器仍会根据框数量差异判断是否补全
                if self.skip_frames > 0 and is_skipped_frame:
                    # 强制为跳过的帧补全：直接使用前后帧的框进行插值
                    if prev_data["boxes"] and next_data["boxes"]:
                        new_boxes, new_detections, was_interp = self.interpolator.interpolate_frame(
                            prev_data["boxes"], curr_data["boxes"], next_data["boxes"],
                            prev_data["detections"], curr_data["detections"], next_data["detections"]
                        )
                        curr_data["boxes"] = new_boxes
                        curr_data["detections"] = new_detections

                        if self.verbose and was_interp:
                            print(f"Frame {curr_data['frame_id']}: 跳帧补全 - 补全了 {len(new_boxes) - len(curr_data.get('boxes', []))} 个框")
                else:
                    # 正常的插值逻辑（根据框数量差异判断）
                    new_boxes, new_detections, was_interp = self.interpolator.interpolate_frame(
                        prev_data["boxes"], curr_data["boxes"], next_data["boxes"],
                        prev_data["detections"], curr_data["detections"], next_data["detections"]
                    )
                    if was_interp:
                        curr_data["boxes"] = new_boxes
                        curr_data["detections"] = new_detections

                        if self.verbose:
                            print(f"Frame {curr_data['frame_id']}: 插值补全 - 补全了框数量")

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
    """示例：使用 VideoOCR 处理视频"""

    # 创建 VideoOCR 实例
    ocr = VideoOCR(
        # ===== OCR 引擎参数 =====
        detect_only=False,
        confidence_threshold=0.85,
        scale_factor=1.0,
        enhance_mode="sharpen",
        use_cuda=True,
        rec_batch_num=6,  # 识别批处理大小

        # ===== 渲染参数 =====
        box_style="red_hollow",

        # ===== 批处理参数 =====
        batch_size=8,
        skip_frames=0,

        # ===== 输出参数 =====
        verbose=True,
    )

    # 处理视频
    ocr.process(
        input_path="custom_nodes\\comfyui_subtitle_detector\\test.mp4",
        max_frames=None,
    )


if __name__ == "__main__":
    main()
