#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 VideoOCR 类的不同OCR模式
展示如何使用统一的API处理不同的OCR需求
"""

from video_ocr import VideoOCR, OCRMode, ModelType, EnhanceMode, BoxStyle


def test_detect_only_mode():
    """模式1: 仅检测（最快，用于生成遮罩）"""
    print("\n" + "="*60)
    print("【模式1】仅检测模式 - 最快，用于生成字幕遮罩")
    print("="*60)

    ocr = VideoOCR(
        # OCR模式
        ocr_mode=OCRMode.DETECT_ONLY,  # 或 "detect_only"

        # 引擎参数
        confidence_threshold=0.3,
        scale_factor=0.8,
        enhance_mode=EnhanceMode.SHARPEN,  # 或 "sharpen"
        use_cuda=True,
        model_type=ModelType.MOBILE,  # 或 "MOBILE" - 快速模型

        # 处理参数
        box_style=BoxStyle.RED_HOLLOW,  # 或 "red_hollow"
        skip_frames=0,
        verbose=True,
    )

    ocr.process(
        input_path="custom_nodes\\comfyui_subtitle_detector\\test.mp4"
    )


def test_detect_rec_mode():
    """模式2: 检测+识别（推荐，用于提取字幕）"""
    print("\n" + "="*60)
    print("【模式2】检测+识别模式 - 推荐，用于提取字幕文字")
    print("="*60)

    ocr = VideoOCR(
        # OCR模式
        ocr_mode=OCRMode.DETECT_REC,  # 或 "detect_rec"

        # 引擎参数
        confidence_threshold=0.5,  # 识别模式用更高阈值
        scale_factor=1.0,
        enhance_mode=EnhanceMode.SHARPEN,  # 或 "sharpen"
        use_cuda=True,
        model_type=ModelType.SERVER,  # 或 "SERVER" - 高精度模型
        rec_batch_num=6,

        # 处理参数
        box_style=BoxStyle.RED_HOLLOW,  # 或 "red_hollow"
        skip_frames=0,
        verbose=True,
    )

    ocr.process(
        input_path="custom_nodes\\comfyui_subtitle_detector\\test.mp4",
        max_frames=100
    )


def test_full_ocr_mode():
    """模式3: 完整OCR（含方向分类，最准确但最慢）"""
    print("\n" + "="*60)
    print("【模式3】完整OCR模式 - 包含方向分类，最准确")
    print("="*60)

    ocr = VideoOCR(
        # OCR模式
        ocr_mode=OCRMode.FULL,  # 或 "full"

        # 引擎参数
        confidence_threshold=0.5,
        scale_factor=1.0,
        enhance_mode=EnhanceMode.SHARPEN,  # 或 "sharpen"
        use_cuda=True,
        model_type=ModelType.SERVER,  # 或 "SERVER"
        rec_batch_num=6,

        # 处理参数
        box_style=BoxStyle.RED_HOLLOW,  # 或 "red_hollow"
        skip_frames=0,
        verbose=True,
    )

    ocr.process(
        input_path="custom_nodes\\comfyui_subtitle_detector\\test.mp4",
        max_frames=50  # 完整OCR较慢，只处理50帧测试
    )


def test_custom_mode():
    """模式4: 自定义模式（手动指定各阶段）"""
    print("\n" + "="*60)
    print("【模式4】自定义模式 - 手动控制检测/分类/识别")
    print("="*60)

    ocr = VideoOCR(
        # OCR模式（自定义）
        ocr_mode=OCRMode.CUSTOM,  # 或 "custom"
        use_det=True,   # 启用检测
        use_cls=False,  # 禁用方向分类
        use_rec=True,   # 启用识别

        # 引擎参数
        confidence_threshold=0.5,
        scale_factor=0.8,
        enhance_mode=EnhanceMode.SHARPEN,  # 或 "sharpen"
        use_cuda=True,
        model_type=ModelType.MOBILE,  # 或 "MOBILE"
        rec_batch_num=6,

        # 处理参数
        box_style=BoxStyle.GREEN_FILL,  # 或 "green_fill"
        skip_frames=0,
        verbose=True,
    )

    ocr.process(
        input_path="custom_nodes\\comfyui_subtitle_detector\\test.mp4",
        max_frames=100
    )


def test_string_mode():
    """使用字符串指定模式（更简洁，仍然支持）"""
    print("\n" + "="*60)
    print("【简洁用法】使用字符串指定模式（向后兼容）")
    print("="*60)

    ocr = VideoOCR(
        ocr_mode="detect_only",      # 字符串模式（仍然支持）
        confidence_threshold=0.3,
        scale_factor=0.8,
        enhance_mode="sharpen",      # 字符串增强模式
        model_type="MOBILE",         # 字符串模型类型
        box_style="red_hollow",      # 字符串框样式
        verbose=True,
    )

    ocr.process(
        input_path="custom_nodes\\comfyui_subtitle_detector\\test.mp4",
        max_frames=100
    )


def main():
    """运行测试"""
    print("="*60)
    print("VideoOCR 多模式测试脚本")
    print("="*60)

    # 默认运行仅检测模式（最常用）
    test_detect_only_mode()

    # 取消注释以测试其他模式
    # test_detect_rec_mode()      # 检测+识别
    # test_full_ocr_mode()        # 完整OCR（含方向分类）
    # test_custom_mode()          # 自定义模式
    # test_string_mode()          # 字符串模式

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
