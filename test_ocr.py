#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频字幕检测测试脚本 - 独立运行，不依赖 ComfyUI

用法:
    python test_ocr.py <video_path> [options]

示例:
    # 基本测试 - 检测第一帧
    python test_ocr.py input.mp4

    # 检测前10帧并保存标注图像
    python test_ocr.py input.mp4 --frames 10 --save

    # 使用GPU加速
    python test_ocr.py input.mp4 --cuda

    # 调整置信度阈值
    python test_ocr.py input.mp4 --threshold 0.2

    # 检测并识别文字内容
    python test_ocr.py input.mp4 --recognize
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from rapid_ocr_engine import RapidOCREngine
from frame_renderer import FrameRenderer


def load_video_frames(video_path: str, max_frames: int = 1, skip_frames: int = 0) -> list:
    """加载视频帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    frames = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
    print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # 跳过前N帧
    for _ in range(skip_frames):
        cap.read()
        frame_idx += 1

    # 读取帧
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame_idx, frame))
        frame_idx += 1

    cap.release()
    print(f"已加载 {len(frames)} 帧 (从第 {skip_frames} 帧开始)")
    return frames


def draw_boxes(frame: np.ndarray, detections: list, style: str = "red_hollow") -> np.ndarray:
    """在帧上绘制检测框"""
    renderer = FrameRenderer(box_style=style)
    boxes = [det["bbox"] for det in detections]
    result_frame, _ = renderer.draw_boxes(frame, boxes)
    return result_frame


def main():
    parser = argparse.ArgumentParser(description="视频字幕检测测试")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("--frames", "-n", type=int, default=1, help="检测帧数 (默认: 1)")
    parser.add_argument("--skip", "-s", type=int, default=0, help="跳过前N帧 (默认: 0)")
    parser.add_argument("--threshold", "-t", type=float, default=0.3, help="置信度阈值 (默认: 0.3)")
    parser.add_argument("--scale", type=float, default=0.8, help="缩放因子 (默认: 0.8)")
    parser.add_argument("--enhance", "-e", choices=["none", "clahe", "sharpen", "denoise", "denoise_sharpen"],
                        default="sharpen", help="图像增强模式 (默认: sharpen)")
    parser.add_argument("--model", "-m", choices=["MOBILE", "SERVER"], default="MOBILE",
                        help="模型类型 (默认: MOBILE)")
    parser.add_argument("--cuda", action="store_true", help="使用GPU加速")
    parser.add_argument("--recognize", "-r", action="store_true", help="识别文字内容 (默认只检测位置)")
    parser.add_argument("--save", action="store_true", help="保存标注后的图像")
    parser.add_argument("--output", "-o", type=str, default="ocr_output", help="输出目录 (默认: ocr_output)")
    parser.add_argument("--style", choices=["red_hollow", "green_hollow", "blue_hollow", "red_filled"],
                        default="red_hollow", help="标注样式 (默认: red_hollow)")

    args = parser.parse_args()

    # 检查视频文件
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)

    # 初始化 OCR 引擎
    print("\n" + "=" * 50)
    print("初始化 RapidOCR 引擎...")
    print(f"  模式: {'检测+识别' if args.recognize else '仅检测'}")
    print(f"  模型: {args.model}")
    print(f"  GPU: {'启用' if args.cuda else '禁用'}")
    print(f"  置信度阈值: {args.threshold}")
    print(f"  缩放因子: {args.scale}")
    print(f"  图像增强: {args.enhance}")
    print("=" * 50 + "\n")

    enhance_mode = None if args.enhance == "none" else args.enhance

    engine = RapidOCREngine(
        detect_only=not args.recognize,
        confidence_threshold=args.threshold,
        scale_factor=args.scale,
        enhance_mode=enhance_mode,
        use_cuda=args.cuda,
        model_type=args.model,
    )

    # 加载视频帧
    print("加载视频帧...")
    frames = load_video_frames(args.video, max_frames=args.frames, skip_frames=args.skip)

    if not frames:
        print("错误: 无法读取视频帧")
        sys.exit(1)

    # 创建输出目录
    if args.save:
        os.makedirs(args.output, exist_ok=True)
        print(f"输出目录: {args.output}")

    # 处理每一帧
    print("\n开始检测...")
    total_time = 0
    total_detections = 0

    for frame_idx, frame in frames:
        print(f"\n--- 帧 {frame_idx} ---")

        # 计时
        start_time = time.time()
        boxes, detections = engine.process_frame(frame)
        elapsed = time.time() - start_time
        total_time += elapsed

        print(f"检测耗时: {elapsed:.3f}s")
        print(f"检测到 {len(detections)} 个文字区域")

        total_detections += len(detections)

        # 打印检测结果
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det["confidence"]
            text = det.get("text", "")

            # 计算边界框的外接矩形
            pts = np.array(bbox)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)

            if text:
                print(f"  [{i+1}] 位置: ({x_min:.0f},{y_min:.0f})-({x_max:.0f},{y_max:.0f}), "
                      f"置信度: {conf:.3f}, 文字: {text}")
            else:
                print(f"  [{i+1}] 位置: ({x_min:.0f},{y_min:.0f})-({x_max:.0f},{y_max:.0f}), "
                      f"置信度: {conf:.3f}")

        # 保存标注图像
        if args.save and detections:
            annotated = draw_boxes(frame.copy(), detections, style=args.style)
            output_path = os.path.join(args.output, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(output_path, annotated)
            print(f"  已保存: {output_path}")

    # 统计信息
    print("\n" + "=" * 50)
    print("检测完成!")
    print(f"  处理帧数: {len(frames)}")
    print(f"  总检测数: {total_detections}")
    print(f"  总耗时: {total_time:.3f}s")
    if len(frames) > 0:
        print(f"  平均每帧: {total_time/len(frames):.3f}s ({len(frames)/total_time:.1f} FPS)")
    print("=" * 50)


if __name__ == "__main__":
    main()
