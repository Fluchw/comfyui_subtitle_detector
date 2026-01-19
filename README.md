# ComfyUI Subtitle Detector & Eraser

ComfyUI 字幕检测与擦除节点，支持视频字幕的自动检测和智能擦除。

## 功能特性

- **字幕检测** - 使用 RapidOCR 进行高精度文字检测，支持 GPU 加速
- **字幕擦除** - 使用 ProPainter 进行视频修复，基于光流的智能补全
- **流式处理** - 支持长视频分段处理，优化内存使用
- **实时预览** - 支持检测结果和擦除结果的实时预览

## 节点说明

### SubtitleDetectorRapidOCR

字幕检测节点，输出标注图像和字幕 mask。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| ocr_mode | OCR 模式 (detect_only / detect_and_recognize) | detect_only |
| model_type | 模型类型 (MOBILE / SERVER) | MOBILE |
| confidence_threshold | 置信度阈值 | 0.3 |
| scale_factor | 缩放因子 | 0.8 |
| enhance_mode | 图像增强模式 | sharpen |
| box_style | 标注样式 | red_hollow |
| use_cuda | GPU 加速 | True |

### SubtitleEraserProPainter

字幕擦除节点，使用 ProPainter 进行视频修复。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| propainter_model | ProPainter 模型文件 | ProPainter.pth |
| raft_model | RAFT 光流模型 | raft-things.pth |
| flow_model | 流补全模型 | recurrent_flow_completion.pth |
| mask_dilation | mask 膨胀 | 4 |
| ref_stride | 参考帧间隔 | 10 |
| neighbor_length | 邻近帧数量 | 10 |
| subvideo_length | 子视频长度 | 80 |

## 安装

### 1. 克隆仓库

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/comfyui_subtitle_detector.git
```

### 2. 安装依赖

```bash
cd comfyui_subtitle_detector
pip install -r requirements.txt
```

### 3. 下载模型

将以下模型文件放入 `ComfyUI/models/DiffuEraser/` 目录：

- `ProPainter.pth` - ProPainter 主模型
- `raft-things.pth` - RAFT 光流模型
- `recurrent_flow_completion.pth` - 流补全模型

模型下载地址：[ProPainter Releases](https://github.com/sczhou/ProPainter/releases)

## 使用示例

### 基本工作流

```
VHS_LoadVideo → SubtitleDetectorRapidOCR → SubtitleEraserProPainter → VHS_VideoCombine
```

1. 使用 `VHS_LoadVideo` 加载视频
2. 使用 `SubtitleDetectorRapidOCR` 检测字幕区域
3. 使用 `SubtitleEraserProPainter` 擦除字幕
4. 使用 `VHS_VideoCombine` 保存视频

示例工作流文件位于 `workflows/subtitle_eraser_workflow.json`

## 依赖

```
rapidocr>=3.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
scipy>=1.10.0
tqdm>=4.65.0
psutil>=5.9.0
```

## 致谢

本项目的实现参考并使用了以下优秀的开源项目：

- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** - 高性能 OCR 引擎 (Apache-2.0)
- **[ComfyUI_DiffuEraser](https://github.com/smthemex/ComfyUI_DiffuEraser)** - ProPainter/DiffuEraser 的 ComfyUI 集成参考
- **[DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser)** - 基于扩散模型的视频修复 (Apache-2.0)
- **[ProPainter](https://github.com/sczhou/ProPainter)** - 基于光流的视频修复模型 (NTU S-Lab License 1.0)

## 许可证

本项目代码采用 Apache-2.0 许可证。

**重要提示**：本项目使用了 [ProPainter](https://github.com/sczhou/ProPainter)，其采用 NTU S-Lab License 1.0 许可证，**仅限非商业用途**。如需商业使用，请联系 ProPainter 作者获取授权。

## 贡献

欢迎提交 Issue 和 Pull Request！
