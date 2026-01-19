#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Subtitle Eraser Node - 使用 ProPainter + DiffuEraser 进行字幕擦除
独立实现，不依赖外部插件
"""

import os
import gc
import copy
import torch
import numpy as np
from PIL import Image

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ComfyUI imports
import folder_paths

# 进度条支持
try:
    from comfy.utils import ProgressBar
    HAS_PROGRESS_BAR = True
except ImportError:
    HAS_PROGRESS_BAR = False

# 本地导入
from .propainter.inference import Propainter
from .propainter.model.misc import get_device
from .libs.diffueraser import DiffuEraser


def tensor_to_pil_list(images_tensor, width=None, height=None, start_idx=0, end_idx=None):
    """将 ComfyUI tensor [B,H,W,C] 转换为 PIL Image 列表 - 支持分段处理"""
    if end_idx is None:
        end_idx = images_tensor.shape[0]

    pil_list = []
    for i in range(start_idx, end_idx):
        img_np = (images_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        if width and height:
            pil_img = pil_img.resize((width, height), Image.LANCZOS)
        pil_list.append(pil_img)
    return pil_list


def mask_tensor_to_pil_list(mask_tensor, width=None, height=None, start_idx=0, end_idx=None):
    """将 ComfyUI mask tensor [B,H,W] 转换为 PIL Image 列表（灰度） - 支持分段处理"""
    if end_idx is None:
        end_idx = mask_tensor.shape[0]

    pil_list = []
    for i in range(start_idx, end_idx):
        mask_np = (mask_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(mask_np, mode='L')
        if width and height:
            pil_img = pil_img.resize((width, height), Image.NEAREST)
        # 转换为 RGB 以匹配期望格式
        pil_img = pil_img.convert('RGB')
        pil_list.append(pil_img)
    return pil_list


def pil_list_to_tensor(pil_list):
    """将 PIL Image 列表转换为 ComfyUI tensor [B,H,W,C] - 流式处理避免内存溢出"""
    if len(pil_list) == 0:
        return torch.empty(0)

    # 获取尺寸
    first_img = np.array(pil_list[0])
    h, w = first_img.shape[:2]
    c = first_img.shape[2] if len(first_img.shape) > 2 else 1

    # 预分配 tensor（在 CPU 上）
    result = torch.zeros((len(pil_list), h, w, c), dtype=torch.float32)

    # 逐帧转换
    for i, pil_img in enumerate(pil_list):
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        if len(img_np.shape) == 2:
            img_np = img_np[:, :, np.newaxis]
        result[i] = torch.from_numpy(img_np)

    return result


# ===== 模型路径设置 =====
DiffuEraser_weights_path = os.path.join(folder_paths.models_dir, "DiffuEraser")
if not os.path.exists(DiffuEraser_weights_path):
    os.makedirs(DiffuEraser_weights_path)

# 当前节点路径
current_node_path = os.path.dirname(os.path.abspath(__file__))


class SubtitleEraserProPainter:
    """
    ProPainter 视频补全节点 - 基于光流的快速视频修复

    输入检测到的字幕 mask，输出修复后的视频
    """

    # 类级别缓存
    _cached_model = None
    _cached_model_paths = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用的模型文件列表
        model_files = ["none"]
        if os.path.exists(DiffuEraser_weights_path):
            files = os.listdir(DiffuEraser_weights_path)
            model_files += [f for f in files if f.endswith(('.pth', '.pt', '.safetensors'))]

        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),

                # ProPainter 模型文件
                "propainter_model": (model_files, {"default": "ProPainter.pth" if "ProPainter.pth" in model_files else "none"}),
                "raft_model": (model_files, {"default": "raft-things.pth" if "raft-things.pth" in model_files else "none"}),
                "flow_model": (model_files, {"default": "recurrent_flow_completion.pth" if "recurrent_flow_completion.pth" in model_files else "none"}),

                # 处理参数
                "mask_dilation": ("INT", {"default": 4, "min": 0, "max": 20, "step": 1}),
                "ref_stride": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "neighbor_length": ("INT", {"default": 10, "min": 2, "max": 50, "step": 2}),
                "subvideo_length": ("INT", {"default": 80, "min": 10, "max": 200, "step": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("erased_images",)
    FUNCTION = "erase_subtitles"
    CATEGORY = "SubtitleDetector"

    def erase_subtitles(self, images, masks, propainter_model, raft_model, flow_model,
                       mask_dilation, ref_stride, neighbor_length, subvideo_length):

        if propainter_model == "none" or raft_model == "none" or flow_model == "none":
            raise ValueError("Please select all three model files: propainter_model, raft_model, flow_model")

        # 获取模型路径
        propainter_path = os.path.join(DiffuEraser_weights_path, propainter_model)
        raft_path = os.path.join(DiffuEraser_weights_path, raft_model)
        flow_path = os.path.join(DiffuEraser_weights_path, flow_model)

        # 检查模型是否存在
        for path, name in [(propainter_path, "ProPainter"), (raft_path, "RAFT"), (flow_path, "Flow")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} model not found: {path}")

        device = get_device()
        print(f"[SubtitleEraser] Using device: {device}")

        # 检查缓存的模型
        current_paths = (raft_path, flow_path, propainter_path)
        if SubtitleEraserProPainter._cached_model is not None and SubtitleEraserProPainter._cached_model_paths == current_paths:
            print("[SubtitleEraser] Using cached ProPainter model")
            model = SubtitleEraserProPainter._cached_model
        else:
            print("[SubtitleEraser] Loading ProPainter model directly to GPU...")
            model = Propainter(device=device)  # 直接加载到 GPU
            model.load_propainter(raft_path, flow_path, propainter_path)
            SubtitleEraserProPainter._cached_model = model
            SubtitleEraserProPainter._cached_model_paths = current_paths

        # 准备输入数据
        batch_size = images.shape[0]
        h, w = images.shape[1], images.shape[2]

        # 确保尺寸是8的倍数
        new_h = h - h % 8
        new_w = w - w % 8

        print(f"[SubtitleEraser] Processing {batch_size} frames at {new_w}x{new_h}")

        # ===== 内存优化：真正的流式处理，不在内存中保留所有帧 =====
        # 分段处理参数 - 根据分辨率动态调整（更保守的设置）
        pixels_per_frame = new_w * new_h
        if pixels_per_frame > 1920 * 1080:
            chunk_size = min(subvideo_length, 8)   # 超高清：最多8帧
        elif pixels_per_frame > 1280 * 720:
            chunk_size = min(subvideo_length, 10)  # 高清（1080p）：最多10帧
        elif pixels_per_frame > 640 * 480:
            chunk_size = min(subvideo_length, 16)  # 720p：最多16帧
        else:
            chunk_size = min(subvideo_length, 24)  # 标清：最多24帧

        overlap = min(neighbor_length, 2)  # 减少重叠帧数

        # 计算 chunk 数量
        if batch_size <= chunk_size:
            num_chunks = 1
        else:
            num_chunks = (batch_size + chunk_size - overlap - 1) // (chunk_size - overlap)

        print(f"[SubtitleEraser] Will process in {num_chunks} chunks (chunk_size={chunk_size}, overlap={overlap})")

        # 进度条
        pbar = ProgressBar(batch_size) if HAS_PROGRESS_BAR else None

        # 输出 tensor 会在第一个 chunk 处理后根据实际输出尺寸创建
        output_tensor = None
        actual_out_h = None
        actual_out_w = None

        try:
            processed_idx = 0
            chunk_idx = 0

            while processed_idx < batch_size:
                # 计算当前分段范围
                start_idx = max(0, processed_idx - overlap) if chunk_idx > 0 else 0
                end_idx = min(batch_size, start_idx + chunk_size)
                chunk_len = end_idx - start_idx

                print(f"[SubtitleEraser] Processing chunk {chunk_idx + 1}/{num_chunks}: frames {start_idx}-{end_idx}")

                # ===== 只加载当前 chunk 需要的帧 =====
                video_pil = []
                for i in range(start_idx, end_idx):
                    img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    if new_w != w or new_h != h:
                        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                    video_pil.append(pil_img)

                mask_pil = []
                for i in range(start_idx, end_idx):
                    mask_np = (masks[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(mask_np, mode='L')
                    if new_w != w or new_h != h:
                        pil_img = pil_img.resize((new_w, new_h), Image.NEAREST)
                    pil_img = pil_img.convert('RGB')
                    mask_pil.append(pil_img)

                # 处理当前分段
                result_pil = model.forward(
                    video=video_pil,
                    mask=mask_pil,
                    load_videobypath=False,
                    video_length=chunk_len,
                    height=new_h,
                    width=new_w,
                    mask_dilation=mask_dilation,
                    ref_stride=ref_stride,
                    neighbor_length=min(neighbor_length, chunk_len - 1),
                    subvideo_length=min(subvideo_length, chunk_len),
                    save_fps=24.0
                )

                # 第一个 chunk：根据实际输出尺寸创建 tensor
                if output_tensor is None and len(result_pil) > 0:
                    first_result = np.array(result_pil[0])
                    actual_out_h, actual_out_w = first_result.shape[0], first_result.shape[1]
                    print(f"[SubtitleEraser] ProPainter output size: {actual_out_w}x{actual_out_h}")
                    output_tensor = torch.zeros((batch_size, actual_out_h, actual_out_w, 3), dtype=torch.float32)

                # 立即将结果写入输出 tensor（跳过重叠部分）
                write_start = overlap if chunk_idx > 0 else 0
                for i, pil_img in enumerate(result_pil):
                    if i >= write_start:
                        out_idx = start_idx + i
                        if out_idx < batch_size:
                            img_np = np.array(pil_img).astype(np.float32) / 255.0
                            output_tensor[out_idx] = torch.from_numpy(img_np)
                            if pbar:
                                pbar.update(1)

                # 立即释放当前 chunk 的内存
                del video_pil, mask_pil, result_pil

                # 强制垃圾回收
                gc.collect()
                gc.collect()  # 多次调用确保释放
                torch.cuda.empty_cache()

                # 打印内存状态
                if HAS_PSUTIL:
                    mem = psutil.virtual_memory()
                    print(f"[SubtitleEraser] Memory: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB ({mem.percent}%)")

                # 更新进度
                processed_idx = end_idx
                chunk_idx += 1

            # 释放输入 tensor
            del images, masks
            gc.collect()
            gc.collect()

            # 释放显存
            gc.collect()
            torch.cuda.empty_cache()

            # 如果输出尺寸与原始尺寸不同，调整回原始尺寸
            if actual_out_h != h or actual_out_w != w:
                print(f"[SubtitleEraser] Resizing from {actual_out_w}x{actual_out_h} to original {w}x{h}...")
                resized_tensor = torch.zeros((batch_size, h, w, 3), dtype=torch.float32)
                for i in range(batch_size):
                    pil_img = Image.fromarray((output_tensor[i].numpy() * 255).astype(np.uint8))
                    pil_img = pil_img.resize((w, h), Image.LANCZOS)
                    resized_tensor[i] = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
                    output_tensor[i] = 0  # 释放旧帧内存
                    if i % 50 == 0:
                        gc.collect()
                output_tensor = resized_tensor
                gc.collect()

            print(f"[SubtitleEraser] Done! Output shape: {output_tensor.shape}")
            return (output_tensor,)

        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            raise e


class SubtitleEraserDiffuEraser:
    """
    DiffuEraser 精修节点 - 基于扩散模型的高质量视频修复

    需要先经过 ProPainter 处理生成 priori，再使用本节点精修
    """

    # 类级别缓存
    _cached_model = None
    _cached_model_config = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 VAE 和 LoRA 文件
        vae_files = ["none"] + folder_paths.get_filename_list("vae")
        lora_files = ["none"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                "images": ("IMAGE",),  # 原始图像
                "masks": ("MASK",),    # 字幕 mask
                "priori_images": ("IMAGE",),  # ProPainter 输出的 priori
                "positive": ("CONDITIONING",),  # 文本编码

                # 模型选择
                "vae": (vae_files, {"default": "none"}),
                "lora": (lora_files, {"default": "none"}),

                # 生成参数
                "steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                # 高级参数
                "mask_dilation": ("INT", {"default": 4, "min": 0, "max": 20, "step": 1}),
                "blended": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("erased_images",)
    FUNCTION = "refine_erase"
    CATEGORY = "SubtitleDetector"

    def refine_erase(self, images, masks, priori_images, positive, vae, lora,
                    steps, seed, mask_dilation, blended):

        device = get_device()
        print(f"[SubtitleEraser DiffuEraser] Using device: {device}")

        # 获取模型路径
        sd_repo = os.path.join(current_node_path, "sd15_repo")
        original_config = os.path.join(current_node_path, "libs/v1-inference.yaml")

        vae_path = folder_paths.get_full_path("vae", vae) if vae != "none" else None
        lora_path = folder_paths.get_full_path("loras", lora) if lora != "none" else None

        if vae_path is None:
            raise ValueError("Please select a VAE model")

        # 检查缓存
        current_config = (vae_path, lora_path)
        if SubtitleEraserDiffuEraser._cached_model is not None and SubtitleEraserDiffuEraser._cached_model_config == current_config:
            print("[SubtitleEraser] Using cached DiffuEraser model")
            model = SubtitleEraserDiffuEraser._cached_model
        else:
            print("[SubtitleEraser] Loading DiffuEraser model...")
            model = DiffuEraser(device)
            model.load_model(sd_repo, DiffuEraser_weights_path, vae_path, original_config, lora_path if lora_path else "")
            SubtitleEraserDiffuEraser._cached_model = model
            SubtitleEraserDiffuEraser._cached_model_config = current_config

        # 准备数据
        batch_size = images.shape[0]
        h, w = images.shape[1], images.shape[2]
        new_h = h - h % 8
        new_w = w - w % 8

        print(f"[SubtitleEraser DiffuEraser] Processing {batch_size} frames at {new_w}x{new_h}")

        # 转换为 PIL
        video_pil = tensor_to_pil_list(images, new_w, new_h)
        mask_pil = mask_tensor_to_pil_list(masks, new_w, new_h)
        priori_pil = tensor_to_pil_list(priori_images, new_w, new_h)

        pbar = ProgressBar(1) if HAS_PROGRESS_BAR else None

        try:
            model.to(device)

            # 运行 DiffuEraser
            result_pil = model.forward(
                validation_image=video_pil,
                validation_mask=mask_pil,
                prioris=priori_pil,
                output_path=folder_paths.get_output_directory(),
                positive=positive,
                load_videobypath=False,
                max_img_size=1920,
                video_length=batch_size,
                mask_dilation_iter=mask_dilation,
                seed=seed,
                blended=blended,
                num_inference_steps=steps,
                fps=24,
                img_size=(new_w, new_h),
                if_save_video=False
            )

            gc.collect()
            torch.cuda.empty_cache()

            if pbar:
                pbar.update(1)

            # 转换回 tensor
            result_tensor = pil_list_to_tensor(result_pil)

            if new_h != h or new_w != w:
                result_pil_resized = [img.resize((w, h), Image.LANCZOS) for img in result_pil]
                result_tensor = pil_list_to_tensor(result_pil_resized)

            print(f"[SubtitleEraser DiffuEraser] Done! Output shape: {result_tensor.shape}")
            return (result_tensor,)

        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            raise e


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "SubtitleEraserProPainter": SubtitleEraserProPainter,
    "SubtitleEraserDiffuEraser": SubtitleEraserDiffuEraser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleEraserProPainter": "Subtitle Eraser (ProPainter)",
    "SubtitleEraserDiffuEraser": "Subtitle Eraser (DiffuEraser Refine)",
}
