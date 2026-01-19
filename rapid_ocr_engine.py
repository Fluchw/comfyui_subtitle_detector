#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RapidOCR 引擎模块 - 负责文字检测和识别
"""

from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from rapidocr import RapidOCR, EngineType, OCRVersion, LangDet, LangRec, ModelType


class ImageEnhancer:
    """图像增强器 - 优化版，支持多种预处理模式以提升文字检测准确率"""

    # 复用 CLAHE 对象，避免每帧重复创建
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 高对比度 CLAHE，用于 sharpen 模式
    _clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def __init__(self, use_gpu: bool = False):
        """初始化增强器

        Args:
            use_gpu: 是否使用GPU加速图像处理
        """
        self.use_gpu = use_gpu
        self.device = None

        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                    print(f"[ImageEnhancer] 使用 GPU 进行图像增强")
                else:
                    print(f"[ImageEnhancer] GPU 不可用，使用 CPU")
            except ImportError:
                print(f"[ImageEnhancer] PyTorch 未安装，使用 CPU")

    def enhance(self, frame: np.ndarray, mode: Optional[str]) -> np.ndarray:
        """
        图像预处理增强

        Args:
            frame: 原始帧 (BGR)
            mode: 增强模式
                - None: 不处理
                - "clahe": CLAHE 对比度增强（通用场景）
                - "binary": 自适应二值化（高对比度字幕）
                - "both": CLAHE + 二值化
                - "sharpen": CLAHE + 锐化（推荐，提升检测准确率）
                - "denoise": 去噪 + CLAHE（视频压缩噪点多时使用）
                - "denoise_sharpen": 去噪 + CLAHE + 锐化（综合处理）

        Returns:
            增强后的帧 (BGR)
        """
        if not mode:
            return frame

        if mode == "clahe":
            return self._apply_clahe(frame)
        elif mode == "binary":
            return self._apply_binary(frame)
        elif mode == "both":
            return self._apply_clahe_binary(frame)
        elif mode == "sharpen":
            return self._apply_sharpen(frame)
        elif mode == "denoise":
            return self._apply_denoise(frame)
        elif mode == "denoise_sharpen":
            return self._apply_denoise_sharpen(frame)
        else:
            return frame

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """应用 CLAHE 对比度增强"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = ImageEnhancer._clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _apply_binary(self, frame: np.ndarray) -> np.ndarray:
        """应用自适应二值化（保持3通道输出）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _apply_clahe_binary(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE + 二值化（优化：减少颜色空间转换）"""
        # BGR -> LAB，提取 L 通道
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        # CLAHE 增强 L 通道
        l_enhanced = ImageEnhancer._clahe.apply(l)
        # 直接对增强后的 L 通道二值化
        binary = cv2.adaptiveThreshold(
            l_enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _apply_sharpen(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE + 边缘锐化（推荐用于提升检测准确率）"""
        # 如果启用GPU，使用GPU加速
        if self.use_gpu and self.device is not None:
            return self._apply_sharpen_gpu(frame)

        # CPU版本
        # CLAHE 增强对比度
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = ImageEnhancer._clahe_strong.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 锐化核 - 增强边缘
        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
        return sharpened

    def _apply_sharpen_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU加速的锐化处理"""
        import torch
        import torch.nn.functional as F

        # 转换为torch tensor并移到GPU
        img_tensor = torch.from_numpy(frame).to(self.device).float()

        # BGR to LAB (简化版，只处理亮度通道)
        # 提取亮度通道 (简化：使用RGB的加权平均)
        b, g, r = img_tensor[:, :, 0], img_tensor[:, :, 1], img_tensor[:, :, 2]
        l_channel = 0.299 * r + 0.587 * g + 0.114 * b

        # GPU上的对比度增强 (简化版CLAHE效果)
        l_min, l_max = l_channel.min(), l_channel.max()
        if l_max > l_min:
            l_enhanced = (l_channel - l_min) / (l_max - l_min) * 255.0
        else:
            l_enhanced = l_channel

        # 重建图像 (保持色彩比例)
        scale = l_enhanced / (l_channel + 1e-6)
        enhanced = img_tensor * scale.unsqueeze(-1)
        enhanced = torch.clamp(enhanced, 0, 255)

        # GPU上的锐化
        # 重塑为 (1, C, H, W) 格式
        img_4d = enhanced.permute(2, 0, 1).unsqueeze(0) / 255.0

        # 锐化核
        sharpen_kernel = torch.tensor([
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # 对每个通道应用卷积
        channels = []
        for i in range(3):
            channel = img_4d[:, i:i+1, :, :]
            sharpened_channel = F.conv2d(channel, sharpen_kernel, padding=1)
            channels.append(sharpened_channel)

        sharpened = torch.cat(channels, dim=1)
        sharpened = torch.clamp(sharpened * 255.0, 0, 255)

        # 转回numpy
        result = sharpened.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return result

    def _apply_denoise(self, frame: np.ndarray) -> np.ndarray:
        """去噪 + CLAHE（适合视频压缩噪点多的情况）"""
        # 快速去噪（比 fastNlMeansDenoisingColored 更快）
        denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
        # CLAHE
        return self._apply_clahe(denoised)

    def _apply_denoise_sharpen(self, frame: np.ndarray) -> np.ndarray:
        """去噪 + CLAHE + 锐化（综合处理）"""
        # 去噪
        denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
        # CLAHE + 锐化
        return self._apply_sharpen(denoised)


class RapidOCREngine:
    """RapidOCR 引擎 - 负责文字检测和识别"""

    def __init__(
        self,
        detect_only: bool = False,
        confidence_threshold: float = 0.5,
        scale_factor: float = 1.0,
        enhance_mode: Optional[Literal["clahe", "binary", "both", "sharpen", "denoise", "denoise_sharpen"]] = None,
        use_det: bool = True,  # 保留以兼容旧代码
        use_cls: bool = False,  # 保留以兼容旧代码
        use_rec: bool = True,  # 保留以兼容旧代码
        det_limit_side_len: int = 960,  # 保留以兼容旧代码
        det_limit_type: str = "max",  # 保留以兼容旧代码
        use_cuda: bool = False,  # 保留以兼容旧代码,统一使用TORCH引擎
        rec_batch_num: int = 6,  # 识别批处理大小
        model_type: Literal["MOBILE", "SERVER"] = "SERVER",  # 模型类型选择
    ):
        """
        初始化 RapidOCR 引擎

        Args:
            detect_only: 仅检测文字位置，不识别文字内容
            confidence_threshold: 置信度阈值(0-1)
            scale_factor: 缩放因子(0-1)
            enhance_mode: 图像预处理模式
            use_det: 启用文本检测
            use_cls: 启用文本方向分类
            use_rec: 启用文本识别
            det_limit_side_len: 检测边长限制
            det_limit_type: 边长限制类型 ('max' 或 'min')
            use_cuda: (已废弃) 现在统一使用 TORCH 引擎以确保最佳检测效果
            rec_batch_num: 识别批处理大小，推荐值6-12，提升识别阶段速度
            model_type: 模型类型选择
                - "MOBILE": 移动端模型（速度快，准确率中等）
                - "SERVER": 服务器模型（速度慢2-3倍，准确率提升5-10%）
        """
        self.detect_only = detect_only
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.enhance_mode = enhance_mode
        self.model_type = model_type

        # 初始化图像增强器,启用GPU加速
        self.enhancer = ImageEnhancer(use_gpu=use_cuda)

        # 将字符串模型类型转换为 ModelType 枚举
        model_type_enum = ModelType.SERVER if model_type == "SERVER" else ModelType.MOBILE

        # 初始化 RapidOCR (v3.0+)
        # 注意：即使是仅检测模式，也需要配置识别模型参数
        # 实际是否使用识别模型由调用时的use_det/use_cls/use_rec参数控制
        params = {
            "Det.engine_type": EngineType.TORCH,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": model_type_enum,  # 可选模型类型
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.TORCH,
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": model_type_enum,  # 可选模型类型
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.rec_batch_num": rec_batch_num,  # 识别批处理大小
        }

        # 如果 use_cuda=True,配置 TORCH 引擎使用 GPU
        if use_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    # 根据官方文档配置 TORCH GPU 推理
                    params["EngineConfig.torch.use_cuda"] = True
                    params["EngineConfig.torch.gpu_id"] = 0
                    if detect_only:
                        print(f"[RapidOCR] 使用 GPU (仅检测模式): {torch.cuda.get_device_name(0)}")
                    else:
                        print(f"[RapidOCR] 使用 GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("[RapidOCR] CUDA 不可用,使用 CPU")
            except ImportError:
                print("[RapidOCR] PyTorch 未安装,使用 CPU")
        else:
            print("[RapidOCR] 使用 CPU 模式")

        # 注意: 不设置 Global.text_score,使用 RapidOCR 默认值
        # confidence_threshold 将在后处理时手动过滤

        self.model = RapidOCR(params=params)

    def process_frames_batch(self, frames: List[np.ndarray]) -> List[Tuple[List, List]]:
        """
        批量处理多帧(GPU加速)

        Args:
            frames: 输入帧列表 (BGR)

        Returns:
            结果列表,每个元素为 (boxes, detections)
        """
        if not frames:
            return []

        # 如果启用GPU增强,批量处理图像增强
        if self.enhance_mode and self.enhancer.use_gpu and self.enhancer.device is not None:
            enhanced_frames = self._batch_enhance_gpu(frames)
        else:
            # CPU版本逐帧处理
            enhanced_frames = []
            for frame in frames:
                # 先缩放（如果需要）
                if self.scale_factor < 1.0:
                    h, w = frame.shape[:2]
                    new_w = int(w * self.scale_factor)
                    new_h = int(h * self.scale_factor)
                    processed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    processed = frame

                # 再进行图像增强
                if self.enhance_mode:
                    processed = self.enhancer.enhance(processed, self.enhance_mode)

                enhanced_frames.append(processed)

        # OCR处理 - RapidOCR不支持真正的批处理，但可以优化图像预处理流程
        # 注意：这里仍然是逐帧调用OCR模型，因为RapidOCR API限制
        results = []
        for enhanced_frame in enhanced_frames:
            boxes, detections = self._process_single_frame(enhanced_frame)
            results.append((boxes, detections))

        return results

    def _batch_enhance_gpu(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """GPU批量图像增强"""
        import torch
        import torch.nn.functional as F

        # 批量转换为tensor
        batch_tensors = []
        for frame in frames:
            # 缩放处理
            if self.scale_factor < 1.0:
                h, w = frame.shape[:2]
                new_w = int(w * self.scale_factor)
                new_h = int(h * self.scale_factor)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            tensor = torch.from_numpy(frame).float()
            batch_tensors.append(tensor)

        # 堆叠成批
        batch = torch.stack(batch_tensors).to(self.enhancer.device)  # (B, H, W, C)

        # 批量处理 - sharpen
        if self.enhance_mode == "sharpen":
            # 提取亮度通道
            b, g, r = batch[:, :, :, 0], batch[:, :, :, 1], batch[:, :, :, 2]
            l_channel = 0.299 * r + 0.587 * g + 0.114 * b

            # 批量对比度增强
            batch_size = batch.shape[0]
            enhanced_batch = []
            for i in range(batch_size):
                l = l_channel[i]
                img = batch[i]

                l_min, l_max = l.min(), l.max()
                if l_max > l_min:
                    l_enhanced = (l - l_min) / (l_max - l_min) * 255.0
                else:
                    l_enhanced = l

                scale = l_enhanced / (l + 1e-6)
                enhanced = img * scale.unsqueeze(-1)
                enhanced_batch.append(enhanced)

            enhanced_batch = torch.stack(enhanced_batch)
            enhanced_batch = torch.clamp(enhanced_batch, 0, 255)

            # 批量锐化
            batch_4d = enhanced_batch.permute(0, 3, 1, 2) / 255.0  # (B, C, H, W)

            # 锐化核
            sharpen_kernel = torch.tensor([
                [[0, -1, 0],
                 [-1, 5, -1],
                 [0, -1, 0]]
            ], dtype=torch.float32, device=self.enhancer.device).unsqueeze(0)

            # 对每个通道批量卷积
            channels = []
            for i in range(3):
                channel = batch_4d[:, i:i+1, :, :]
                sharpened_channel = F.conv2d(channel, sharpen_kernel, padding=1)
                channels.append(sharpened_channel)

            sharpened_batch = torch.cat(channels, dim=1)
            sharpened_batch = torch.clamp(sharpened_batch * 255.0, 0, 255)

            # 转回numpy
            result_batch = sharpened_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            return [result_batch[i] for i in range(batch_size)]
        else:
            # 其他模式回退到逐帧处理
            results = []
            for frame in frames:
                enhanced = self.enhancer.enhance(frame, self.enhance_mode)
                results.append(enhanced)
            return results

    def process_frame(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        对单帧进行OCR处理

        Args:
            frame: 输入帧 (BGR)

        Returns:
            (boxes, detections)
            boxes: 文字框坐标列表
            detections: 检测结果列表 [{"bbox": [...], "text": "...", "confidence": 0.9}, ...]
        """
        # 先缩放（如果需要）
        if self.scale_factor < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            process_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            process_frame = frame

        # 再进行图像增强
        if self.enhance_mode:
            process_frame = self.enhancer.enhance(process_frame, self.enhance_mode)

        return self._process_single_frame(process_frame)

    def _process_single_frame(self, process_frame: np.ndarray) -> Tuple[List, List]:
        """
        处理单帧的内部方法

        Args:
            process_frame: 已预处理的帧

        Returns:
            (boxes, detections)
        """
        boxes = []
        detections = []

        try:
            # 调用 RapidOCR (v3.0+)
            # 如果是仅检测模式,传递use_det=True, use_cls=False, use_rec=False
            if self.detect_only:
                result = self.model(process_frame, use_det=True, use_cls=False, use_rec=False)
            else:
                result = self.model(process_frame)

            if result is None:
                return boxes, detections

            # 仅检测模式返回 TextDetOutput,包含boxes和scores(检测置信度)
            if self.detect_only:
                # TextDetOutput 对象
                dt_boxes = result.boxes if hasattr(result, 'boxes') and result.boxes is not None else []
                det_scores = result.scores if hasattr(result, 'scores') and result.scores is not None else []

                # 检查是否有检测结果
                has_boxes = len(dt_boxes) > 0 if dt_boxes is not None else False
                if not has_boxes:
                    return boxes, detections

                # 解析结果 - 只有框和检测置信度
                for i, dt_box in enumerate(dt_boxes):
                    # 检测置信度
                    det_confidence = float(det_scores[i]) if i < len(det_scores) else 1.0

                    # 过滤低置信度结果（检测置信度）
                    if det_confidence < self.confidence_threshold:
                        continue

                    # 坐标格式转换
                    box = dt_box.tolist() if hasattr(dt_box, 'tolist') else list(dt_box)

                    # 如果进行了缩放，需要还原坐标
                    if self.scale_factor < 1.0:
                        box = [[p[0] / self.scale_factor, p[1] / self.scale_factor] for p in box]

                    boxes.append(box)
                    detections.append({
                        "bbox": box,
                        "text": "",  # 仅检测模式无文本
                        "confidence": det_confidence  # 检测置信度
                    })
            else:
                # 检测+识别模式返回 RapidOCROutput
                # 尝试从对象属性或字典中提取结果
                if hasattr(result, 'boxes'):
                    # RapidOCROutput 对象 (新版本API)
                    dt_boxes = result.boxes if result.boxes is not None else []
                    rec_texts = result.txts if hasattr(result, 'txts') and result.txts else []
                    rec_scores = result.scores if hasattr(result, 'scores') and result.scores else []
                elif hasattr(result, 'dt_boxes'):
                    # 旧版本 RapidOCROutput 对象
                    dt_boxes = result.dt_boxes if result.dt_boxes else []
                    rec_texts = result.txts if hasattr(result, 'txts') and result.txts else []
                    rec_scores = result.scores if hasattr(result, 'scores') and result.scores else []
                elif isinstance(result, dict):
                    # 字典格式 (兼容)
                    dt_boxes = result.get('dt_boxes', []) or result.get('boxes', [])
                    rec_texts = result.get('txts', [])
                    rec_scores = result.get('scores', [])
                else:
                    return boxes, detections

                # 检查是否有检测结果
                has_boxes = len(dt_boxes) > 0 if dt_boxes is not None else False
                if not has_boxes:
                    return boxes, detections

                # 解析结果 - 包含文本和识别置信度
                for i, dt_box in enumerate(dt_boxes):
                    # 提取文本和置信度
                    text = rec_texts[i] if i < len(rec_texts) else ""
                    confidence = float(rec_scores[i]) if i < len(rec_scores) else 1.0

                    # 手动过滤低置信度结果（识别置信度）
                    if confidence < self.confidence_threshold:
                        continue

                    # 坐标格式转换
                    box = dt_box.tolist() if hasattr(dt_box, 'tolist') else list(dt_box)

                    # 如果进行了缩放，需要还原坐标
                    if self.scale_factor < 1.0:
                        box = [[p[0] / self.scale_factor, p[1] / self.scale_factor] for p in box]

                    boxes.append(box)
                    detections.append({
                        "bbox": box,
                        "text": text,
                        "confidence": confidence  # 识别置信度
                    })

        except Exception as e:
            import traceback
            print(f"RapidOCR processing error: {e}")
            print(traceback.format_exc())
            return [], []

        return boxes, detections

    def get_model_info(self) -> str:
        """获取模型信息描述"""
        mode = "仅检测" if self.detect_only else "检测+识别"
        return f"RapidOCR {mode} (threshold={self.confidence_threshold})"
