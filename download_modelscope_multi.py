#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 ModelScope 下载指定模型的多个子目录或文件。

Usage:
  python download_modelscope_multi.py \
    --model AI-ModelScope/stable-diffusion-v1-5 \
    --subdir text_encoder unet vae \
    --output /mnt/workspace/sd-parts

  # 或下载混合内容：
  python download_modelscope_multi.py \
    --model AI-ModelScope/stable-diffusion-v1-5 \
    --subdir "text_encoder/**" "vae/diffusion_pytorch_model.safetensors" "*.json" \
    --output /mnt/workspace/custom
"""

import argparse
import os
from modelscope import snapshot_download


def normalize_pattern(p):
    """自动为纯目录名补全 /**，保留带通配符或文件扩展名的原样"""
    if '*' in p or '.' in os.path.basename(p.rstrip('/')):
        return p
    # 确保以 / 结尾再加 **
    if not p.endswith('/'):
        p += '/'
    return p + "**"


def main():
    parser = argparse.ArgumentParser(
        description="Download multiple subdirectories or files from a ModelScope model."
    )
    parser.add_argument("--model", required=True, help="Model ID, e.g., AI-ModelScope/stable-diffusion-v1-5")
    parser.add_argument(
        "--subdir",
        nargs='+',  # 接收多个参数
        required=True,
        help="One or more subdirectory names or patterns (e.g., text_encoder unet vae)"
    )
    parser.add_argument("--output", required=True, help="Local output directory")
    parser.add_argument("--revision", default="master", help="Model revision, default: master")

    args = parser.parse_args()

    # 自动标准化每个 pattern
    allow_patterns = [normalize_pattern(p) for p in args.subdir]

    print(f"📥 Model: {args.model}")
    print(f"📂 Patterns: {allow_patterns}")
    print(f"💾 Output: {args.output}")

    try:
        snapshot_download(
            model_id=args.model,
            local_dir=args.output,
            revision=args.revision,
            allow_patterns=allow_patterns
        )
        print("✅ Download completed!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()