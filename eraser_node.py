#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Subtitle Eraser Node - 使用 ProPainter + DiffuEraser 进行字幕擦除
独立实现，不依赖外部插件
"""

import os
import gc
import sys
import copy
import signal
import atexit
import torch
import numpy as np
from PIL import Image

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# 使用 loguru 日志
try:
    from loguru import logger
except ImportError:
    # fallback to print if loguru not available
    class FallbackLogger:
        def info(self, msg): print(f"[INFO] {msg}", flush=True)
        def warning(self, msg): print(f"[WARN] {msg}", flush=True)
        def error(self, msg): print(f"[ERROR] {msg}", flush=True)
        def debug(self, msg): print(f"[DEBUG] {msg}", flush=True)
    logger = FallbackLogger()

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
        # 优化：减少临时数组创建，直接在原地操作
        img_np = images_tensor[i].cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        if width and height:
            pil_img = pil_img.resize((width, height), Image.LANCZOS)
        pil_list.append(pil_img)
        # 显式删除numpy数组，帮助垃圾回收
        del img_np
    return pil_list


def mask_tensor_to_pil_list(mask_tensor, width=None, height=None, start_idx=0, end_idx=None):
    """将 ComfyUI mask tensor [B,H,W] 转换为 PIL Image 列表（灰度） - 支持分段处理"""
    if end_idx is None:
        end_idx = mask_tensor.shape[0]

    pil_list = []
    for i in range(start_idx, end_idx):
        # 优化：减少临时数组创建
        mask_np = mask_tensor[i].cpu().numpy()
        mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(mask_np, mode='L')
        if width and height:
            pil_img = pil_img.resize((width, height), Image.NEAREST)
        # 转换为 RGB 以匹配期望格式
        pil_img_rgb = pil_img.convert('RGB')
        pil_list.append(pil_img_rgb)
        # 显式删除临时对象，帮助垃圾回收
        del mask_np, pil_img
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
                "raft_iter": ("INT", {"default": 20, "min": 1, "max": 40, "step": 1}),
                "fp16": ("BOOLEAN", {"default": True}),
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),  # 0=自动
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("erased_images",)
    FUNCTION = "erase_subtitles"
    CATEGORY = "SubtitleDetector"

    def erase_subtitles(self, images, masks, propainter_model, raft_model, flow_model,
                       mask_dilation, ref_stride, neighbor_length, subvideo_length,
                       raft_iter=20, fp16=True, chunk_size=0):

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
        logger.info(f"[SubtitleEraser] Using device: {device}")

        # 检查缓存的模型
        current_paths = (raft_path, flow_path, propainter_path)
        if SubtitleEraserProPainter._cached_model is not None and SubtitleEraserProPainter._cached_model_paths == current_paths:
            logger.info("[SubtitleEraser] Using cached ProPainter model")
            model = SubtitleEraserProPainter._cached_model
        else:
            logger.info("[SubtitleEraser] Loading ProPainter model directly to GPU...")
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

        logger.info(f"[SubtitleEraser] Processing {batch_size} frames at {new_w}x{new_h}")

        # ===== 内存优化：真正的流式处理，不在内存中保留所有帧 =====
        # 分段处理参数
        if chunk_size > 0:
            # 用户手动指定 chunk_size
            actual_chunk_size = min(subvideo_length, chunk_size)
            logger.info(f"[SubtitleEraser] Using manual chunk_size: {actual_chunk_size}")
        else:
            # 自动根据分辨率动态调整
            pixels_per_frame = new_w * new_h
            if pixels_per_frame > 1920 * 1080:
                actual_chunk_size = min(subvideo_length, 8)   # 超高清：最多8帧
            elif pixels_per_frame > 1280 * 720:
                actual_chunk_size = min(subvideo_length, 10)  # 高清（1080p）：最多10帧
            elif pixels_per_frame > 640 * 480:
                actual_chunk_size = min(subvideo_length, 16)  # 720p：最多16帧
            else:
                actual_chunk_size = min(subvideo_length, 24)  # 标清：最多24帧
            logger.info(f"[SubtitleEraser] Auto chunk_size: {actual_chunk_size} (based on {new_w}x{new_h})")

        overlap = neighbor_length // 2  # 重叠帧数，与原项目一致

        # 计算 chunk 数量
        if batch_size <= actual_chunk_size:
            num_chunks = 1
        else:
            num_chunks = (batch_size + actual_chunk_size - overlap - 1) // (actual_chunk_size - overlap)

        logger.info(f"[SubtitleEraser] Will process in {num_chunks} chunks (chunk_size={actual_chunk_size}, overlap={overlap})")

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
                end_idx = min(batch_size, start_idx + actual_chunk_size)
                chunk_len = end_idx - start_idx

                logger.info(f"[SubtitleEraser] Processing chunk {chunk_idx + 1}/{num_chunks}: frames {start_idx}-{end_idx}")

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
                    raft_iter=raft_iter,
                    fp16=fp16,
                    save_fps=24.0
                )

                # 第一个 chunk：根据实际输出尺寸创建 tensor
                if output_tensor is None and len(result_pil) > 0:
                    first_result = np.array(result_pil[0])
                    actual_out_h, actual_out_w = first_result.shape[0], first_result.shape[1]
                    logger.info(f"[SubtitleEraser] ProPainter output size: {actual_out_w}x{actual_out_h}")
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
                    logger.info(f"[SubtitleEraser] Memory: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB ({mem.percent}%)")

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

            logger.info(f"[SubtitleEraser] Done! Output shape: {output_tensor.shape}")
            return (output_tensor,)

        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            raise e


class SubtitleEraserDiffuEraser:
    """
    DiffuEraser 精修节点 - 基于扩散模型的高质量视频修复

    需要先经过 ProPainter 处理生成 priori，再使用本节点精修

    已移除 SD1.5 完整模型依赖，直接使用 CLIP 和 VAE 模型文件
    """

    # 类级别缓存
    _cached_model = None
    _cached_model_config = None
    _cached_clip_model = None
    _cached_clip_path = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 VAE、CLIP 和 LoRA 文件
        vae_files = ["none"] + folder_paths.get_filename_list("vae")
        clip_files = ["none"] + folder_paths.get_filename_list("clip")
        lora_files = ["none"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                "images": ("IMAGE",),  # 原始图像
                "masks": ("MASK",),    # 字幕 mask
                "priori_images": ("IMAGE",),  # ProPainter 输出的 priori

                # 模型选择 - 不再需要完整 SD1.5 checkpoint
                "vae": (vae_files, {"default": "sd-vae-ft-mse.safetensors" if "sd-vae-ft-mse.safetensors" in vae_files else "none"}),
                "clip": (clip_files, {"default": "clip_l.safetensors" if "clip_l.safetensors" in clip_files else "none"}),
                "lora": (lora_files, {"default": "none"}),

                # 提示词 - 内部处理文本编码
                "prompt": ("STRING", {"default": "clean background, high quality", "multiline": True}),

                # 生成参数
                "steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                # 高级参数
                "mask_dilation": ("INT", {"default": 4, "min": 0, "max": 20, "step": 1}),
                "blended": ("BOOLEAN", {"default": True}),

                # 分块处理参数 - 用于控制显存使用
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1,
                    "tooltip": "每次处理的帧数，0=自动（根据分辨率调整）。DiffuEraser最小需要22帧。"}),
                "chunk_overlap": ("INT", {"default": 8, "min": 4, "max": 20, "step": 1,
                    "tooltip": "分块之间的重叠帧数，用于保证时序一致性"}),

                # Group Offloading 参数 - 用于进一步优化显存
                "unet_group": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1,
                    "tooltip": "UNet 分组卸载块数，越小越省显存但越慢。0=禁用"}),
                "brushnet_group": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1,
                    "tooltip": "BrushNet 分组卸载块数，越小越省显存但越慢。0=禁用"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("erased_images",)
    FUNCTION = "refine_erase"
    CATEGORY = "SubtitleDetector"

    def _load_clip_model(self, clip_path, device):
        """加载 CLIP 模型并缓存"""
        from transformers import CLIPTextModel
        from safetensors.torch import load_file as load_safetensors

        if SubtitleEraserDiffuEraser._cached_clip_model is not None and SubtitleEraserDiffuEraser._cached_clip_path == clip_path:
            logger.info("[SubtitleEraser] Using cached CLIP model")
            return SubtitleEraserDiffuEraser._cached_clip_model

        logger.info(f"[SubtitleEraser] Loading CLIP from {clip_path}...")

        # 加载 CLIP 配置
        sd_repo = os.path.join(current_node_path, "sd15_repo")
        text_encoder_config_path = os.path.join(sd_repo, "text_encoder", "config.json")

        if os.path.exists(text_encoder_config_path):
            # 从本地配置文件加载（只加载配置，不加载权重）
            from transformers import CLIPTextConfig
            config = CLIPTextConfig.from_pretrained(
                os.path.join(sd_repo, "text_encoder"),
                local_files_only=True
            )
            clip_model = CLIPTextModel(config)
        else:
            # 使用默认 CLIP-L 配置
            from transformers import CLIPTextConfig
            config = CLIPTextConfig(
                vocab_size=49408,
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                max_position_embeddings=77,
                hidden_act="quick_gelu",
                layer_norm_eps=1e-5,
                projection_dim=768,
            )
            clip_model = CLIPTextModel(config)

        # 加载权重
        if clip_path.endswith(".safetensors"):
            state_dict = load_safetensors(clip_path)
        else:
            state_dict = torch.load(clip_path, map_location="cpu", weights_only=True)

        # 处理不同格式的 state_dict
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除可能的前缀
            new_key = k
            if k.startswith("text_model."):
                new_key = k
            elif k.startswith("transformer."):
                new_key = k.replace("transformer.", "text_model.")
            elif not k.startswith("text_model"):
                new_key = "text_model." + k
            new_state_dict[new_key] = v

        clip_model.load_state_dict(new_state_dict, strict=False)
        clip_model = clip_model.to(device, dtype=torch.float16)
        clip_model.eval()

        SubtitleEraserDiffuEraser._cached_clip_model = clip_model
        SubtitleEraserDiffuEraser._cached_clip_path = clip_path

        return clip_model

    def _encode_prompt(self, prompt, clip_model, tokenizer, device):
        """使用 CLIP 对提示词进行编码"""
        # Tokenize
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        # Encode
        with torch.no_grad():
            prompt_embeds = clip_model(text_input_ids)[0]

        prompt_embeds = prompt_embeds.to(dtype=torch.float16)

        # 返回格式与 ComfyUI CONDITIONING 兼容: [[embeds, {}]]
        return [[prompt_embeds, {}]]

    def refine_erase(self, images, masks, priori_images, vae, clip, lora, prompt,
                    steps, seed, mask_dilation, blended, chunk_size, chunk_overlap,
                    unet_group, brushnet_group):

        device = get_device()
        logger.info(f"[SubtitleEraser DiffuEraser] Using device: {device}")

        # 获取模型路径
        sd_repo = os.path.join(current_node_path, "sd15_repo")
        original_config = os.path.join(current_node_path, "libs/v1-inference.yaml")

        vae_path = folder_paths.get_full_path("vae", vae) if vae != "none" else None
        clip_path = folder_paths.get_full_path("clip", clip) if clip != "none" else None
        lora_path = folder_paths.get_full_path("loras", lora) if lora != "none" else None

        if vae_path is None:
            raise ValueError("请选择 VAE 模型 (推荐: sd-vae-ft-mse.safetensors)")
        if clip_path is None:
            raise ValueError("请选择 CLIP 模型 (推荐: clip_l.safetensors)")

        # 加载 CLIP 并编码提示词
        clip_model = self._load_clip_model(clip_path, device)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(sd_repo, "tokenizer"),
            local_files_only=True
        )

        positive = self._encode_prompt(prompt, clip_model, tokenizer, device)
        logger.info(f"[SubtitleEraser DiffuEraser] Prompt encoded: '{prompt[:50]}...'")

        # 释放 CLIP 显存 (编码完成后不再需要)
        clip_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # 检查缓存 - 包含 group offloading 参数
        current_config = (vae_path, lora_path, unet_group, brushnet_group)
        if SubtitleEraserDiffuEraser._cached_model is not None and SubtitleEraserDiffuEraser._cached_model_config == current_config:
            logger.info("[SubtitleEraser] Using cached DiffuEraser model")
            model = SubtitleEraserDiffuEraser._cached_model
        else:
            logger.info("[SubtitleEraser] Loading DiffuEraser model...")
            model = DiffuEraser(device)
            model.load_model(sd_repo, DiffuEraser_weights_path, vae_path, original_config, lora_path if lora_path else "")

            # 应用 Group Offloading 优化显存
            if unet_group > 0 or brushnet_group > 0:
                try:
                    from diffusers.hooks import apply_group_offloading
                    logger.info(f"[SubtitleEraser DiffuEraser] Applying group offloading (unet={unet_group}, brushnet={brushnet_group})")

                    if unet_group > 0 and hasattr(model, 'pipeline') and hasattr(model.pipeline, 'unet'):
                        apply_group_offloading(
                            model.pipeline.unet,
                            onload_device=torch.device("cuda"),
                            offload_type="block_level",
                            num_blocks_per_group=unet_group
                        )
                        logger.info(f"[SubtitleEraser DiffuEraser] UNet group offloading applied")

                    if brushnet_group > 0 and hasattr(model, 'pipeline') and hasattr(model.pipeline, 'brushnet'):
                        apply_group_offloading(
                            model.pipeline.brushnet,
                            onload_device=torch.device("cuda"),
                            offload_type="block_level",
                            num_blocks_per_group=brushnet_group
                        )
                        logger.info(f"[SubtitleEraser DiffuEraser] BrushNet group offloading applied")

                except ImportError:
                    logger.warning("[SubtitleEraser DiffuEraser] diffusers.hooks not available, skipping group offloading")
                except Exception as e:
                    logger.warning(f"[SubtitleEraser DiffuEraser] Group offloading failed: {e}")

            # 尝试启用 xformers
            try:
                model.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("[SubtitleEraser DiffuEraser] XFormers memory efficient attention enabled")
            except Exception as e:
                logger.debug(f"[SubtitleEraser DiffuEraser] XFormers not available: {e}")

            SubtitleEraserDiffuEraser._cached_model = model
            SubtitleEraserDiffuEraser._cached_model_config = current_config

        # 准备数据
        batch_size = images.shape[0]
        # 使用 priori_images 的尺寸作为处理尺寸（ProPainter 的输出尺寸）
        # 这样可以避免将低分辨率的 priori upscale 导致质量损失
        h, w = priori_images.shape[1], priori_images.shape[2]
        new_h = h - h % 8
        new_w = w - w % 8

        # DiffuEraser 最小需要 22 帧
        MIN_FRAMES = 22

        # 自动计算 chunk_size (根据分辨率)
        if chunk_size == 0:
            pixels_per_frame = new_w * new_h
            if pixels_per_frame > 1920 * 1080:
                actual_chunk_size = 24  # 超高清
            elif pixels_per_frame > 1280 * 720:
                actual_chunk_size = 32  # 高清 (1080p)
            elif pixels_per_frame > 640 * 480:
                actual_chunk_size = 48  # 720p
            else:
                actual_chunk_size = 64  # 标清
            logger.info(f"[SubtitleEraser DiffuEraser] Auto chunk_size: {actual_chunk_size} (based on {new_w}x{new_h})")
        else:
            actual_chunk_size = max(chunk_size, MIN_FRAMES)
            if chunk_size < MIN_FRAMES:
                logger.info(f"[SubtitleEraser DiffuEraser] Warning: chunk_size increased to {MIN_FRAMES} (DiffuEraser minimum)")

        # 如果总帧数小于最小要求，直接处理
        if batch_size < MIN_FRAMES:
            raise ValueError(f"DiffuEraser 需要至少 {MIN_FRAMES} 帧，当前只有 {batch_size} 帧")

        # 计算分块数量
        overlap = min(chunk_overlap, actual_chunk_size // 4)
        if batch_size <= actual_chunk_size:
            num_chunks = 1
        else:
            num_chunks = max(1, (batch_size - overlap) // (actual_chunk_size - overlap))
            # 确保覆盖所有帧
            if overlap + num_chunks * (actual_chunk_size - overlap) < batch_size:
                num_chunks += 1

        logger.info(f"[SubtitleEraser DiffuEraser] Processing {batch_size} frames at {new_w}x{new_h}")
        logger.info(f"[SubtitleEraser DiffuEraser] Will process in {num_chunks} chunks (chunk_size={actual_chunk_size}, overlap={overlap})")

        # 进度条 - 每个 chunk 一个进度单位
        pbar = ProgressBar(num_chunks) if HAS_PROGRESS_BAR else None

        # 输出 tensor
        output_tensor = torch.zeros((batch_size, h, w, 3), dtype=torch.float32)

        try:
            model.to(device)

            processed_idx = 0
            chunk_idx = 0

            while processed_idx < batch_size:
                # 在每个chunk处理前清理显存，避免累积
                if chunk_idx > 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                # 计算当前分段范围
                start_idx = max(0, processed_idx - overlap) if chunk_idx > 0 else 0
                end_idx = min(batch_size, start_idx + actual_chunk_size)

                # 确保至少有 MIN_FRAMES 帧
                if end_idx - start_idx < MIN_FRAMES:
                    start_idx = max(0, end_idx - MIN_FRAMES)

                chunk_len = end_idx - start_idx

                logger.info(f"[SubtitleEraser DiffuEraser] Processing chunk {chunk_idx + 1}/{num_chunks}: frames {start_idx}-{end_idx} ({chunk_len} frames)")

                # 只转换当前 chunk 需要的帧
                chunk_images = images[start_idx:end_idx]
                chunk_masks = masks[start_idx:end_idx]
                chunk_priori = priori_images[start_idx:end_idx]

                video_pil = tensor_to_pil_list(chunk_images, new_w, new_h)
                mask_pil = mask_tensor_to_pil_list(chunk_masks, new_w, new_h)
                priori_pil = tensor_to_pil_list(chunk_priori, new_w, new_h)

                # 运行 DiffuEraser
                result_pil = model.forward(
                    validation_image=video_pil,
                    validation_mask=mask_pil,
                    prioris=priori_pil,
                    output_path=folder_paths.get_output_directory(),
                    positive=positive,
                    load_videobypath=False,
                    max_img_size=1920,
                    video_length=chunk_len,
                    mask_dilation_iter=mask_dilation,
                    seed=seed,
                    blended=blended,
                    num_inference_steps=steps,
                    fps=24,
                    img_size=(new_w, new_h),
                    if_save_video=False
                )

                # 写入结果 (跳过重叠部分)
                write_start = overlap if chunk_idx > 0 else 0
                for i, pil_img in enumerate(result_pil):
                    if i >= write_start:
                        out_idx = start_idx + i
                        if out_idx < batch_size:
                            # 如果需要调整尺寸
                            if new_h != h or new_w != w:
                                pil_img = pil_img.resize((w, h), Image.LANCZOS)
                            img_np = np.array(pil_img).astype(np.float32) / 255.0
                            output_tensor[out_idx] = torch.from_numpy(img_np)
                            # 立即删除临时numpy数组，避免累积
                            del img_np

                # 更新进度
                if pbar:
                    pbar.update(1)

                # 释放当前 chunk 的内存
                del video_pil, mask_pil, priori_pil, result_pil, chunk_images, chunk_masks, chunk_priori
                # 多次垃圾回收，确保彻底清理
                gc.collect()
                gc.collect()
                torch.cuda.empty_cache()

                # 打印内存状态
                if HAS_PSUTIL:
                    mem = psutil.virtual_memory()
                    logger.info(f"[SubtitleEraser DiffuEraser] Memory: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB ({mem.percent}%)")
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                    logger.info(f"[SubtitleEraser DiffuEraser] GPU Memory Peak: {gpu_mem:.2f}GB")

                # 更新进度
                processed_idx = end_idx
                chunk_idx += 1

            logger.info(f"[SubtitleEraser DiffuEraser] Done! Output shape: {output_tensor.shape}")
            return (output_tensor,)

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


# ==================== 清理函数 ====================
def cleanup_models():
    """清理所有缓存的模型，释放 GPU 资源"""
    logger.info("[SubtitleEraser] Cleaning up cached models...")

    # 清理 ProPainter 模型
    if SubtitleEraserProPainter._cached_model is not None:
        try:
            del SubtitleEraserProPainter._cached_model
            SubtitleEraserProPainter._cached_model = None
            SubtitleEraserProPainter._cached_model_paths = None
        except:
            pass

    # 清理 DiffuEraser 模型
    if SubtitleEraserDiffuEraser._cached_model is not None:
        try:
            del SubtitleEraserDiffuEraser._cached_model
            SubtitleEraserDiffuEraser._cached_model = None
            SubtitleEraserDiffuEraser._cached_model_config = None
        except:
            pass

    # 清理 CLIP 模型
    if SubtitleEraserDiffuEraser._cached_clip_model is not None:
        try:
            del SubtitleEraserDiffuEraser._cached_clip_model
            SubtitleEraserDiffuEraser._cached_clip_model = None
            SubtitleEraserDiffuEraser._cached_clip_path = None
        except:
            pass

    # 强制垃圾回收和清空 CUDA 缓存
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # 移除 synchronize()，避免重启时卡住
            # torch.cuda.synchronize()
        except:
            pass

    logger.info("[SubtitleEraser] Cleanup completed.")


def signal_handler(signum, frame):
    """处理中断信号"""
    try:
        logger.info(f"[SubtitleEraser] Received signal {signum}, cleaning up...")
        cleanup_models()
    except:
        pass
    # 恢复默认处理并重新发送信号
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# 注册退出时清理
atexit.register(cleanup_models)

# 注册信号处理 (仅在主线程中)
try:
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill
except:
    # 可能不在主线程中，忽略
    pass
