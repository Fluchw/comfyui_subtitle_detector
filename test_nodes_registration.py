#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test node registration"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the module
import comfyui_subtitle_detector
NODE_CLASS_MAPPINGS = comfyui_subtitle_detector.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = comfyui_subtitle_detector.NODE_DISPLAY_NAME_MAPPINGS

print("Registered nodes:")
print("=" * 60)
for node_name, node_class in NODE_CLASS_MAPPINGS.items():
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
    print(f"  {node_name}")
    print(f"    Display: {display_name}")
    print(f"    Class: {node_class.__name__}")
    print()

print(f"Total: {len(NODE_CLASS_MAPPINGS)} nodes registered")
print("=" * 60)
