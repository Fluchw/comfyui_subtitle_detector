#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Subtitle Detector - RapidOCR Implementation
"""

import os
import folder_paths

# 注册 ProPainter 相关模型的自定义路径
# 这样用户可以在 extra_model_paths.yaml 中配置这些模型的路径
propainter_models_dir = os.path.join(folder_paths.models_dir, "propainter")
if not os.path.exists(propainter_models_dir):
    os.makedirs(propainter_models_dir)

# 注册三种 ProPainter 模型类型
folder_paths.add_model_folder_path("propainter", propainter_models_dir)
folder_paths.add_model_folder_path("raft", propainter_models_dir)
folder_paths.add_model_folder_path("flow_completion", propainter_models_dir)

# 同时保持对旧的 DiffuEraser 路径的兼容
diffueraser_models_dir = os.path.join(folder_paths.models_dir, "DiffuEraser")
if not os.path.exists(diffueraser_models_dir):
    os.makedirs(diffueraser_models_dir)
folder_paths.add_model_folder_path("propainter", diffueraser_models_dir)
folder_paths.add_model_folder_path("raft", diffueraser_models_dir)
folder_paths.add_model_folder_path("flow_completion", diffueraser_models_dir)

from .nodes import NODE_CLASS_MAPPINGS as NODES_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_MAPPINGS

from .eraser_node import NODE_CLASS_MAPPINGS as ERASER_MAPPINGS
from .eraser_node import NODE_DISPLAY_NAME_MAPPINGS as ERASER_DISPLAY_MAPPINGS

# 合并所有节点
NODE_CLASS_MAPPINGS = {**NODES_MAPPINGS, **ERASER_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY_MAPPINGS, **ERASER_DISPLAY_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS',]
