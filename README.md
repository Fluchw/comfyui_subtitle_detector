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

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| propainter_model | - | ProPainter.pth | ProPainter 主模型文件 |
| raft_model | - | raft-things.pth | RAFT 光流模型 |
| flow_model | - | recurrent_flow_completion.pth | 流补全模型 |
| mask_dilation | 0-20 | 4 | mask 膨胀像素，扩大擦除区域边缘 |
| ref_stride | 1-50 | 10 | 参考帧步长，越小质量越好但速度越慢 |
| neighbor_length | 2-50 | 10 | 邻近帧数量，用于时序一致性 |
| subvideo_length | 10-200 | 80 | 子视频长度，长视频会被分段处理 |
| raft_iter | 1-40 | 20 | RAFT 光流迭代次数，越高精度越好但越慢 |
| fp16 | bool | True | 半精度推理，开启可节省显存 |
| chunk_size | 0-100 | 0 | 每次处理帧数，0=根据分辨率自动调整 |

#### chunk_size 参数说明

- **chunk_size = 0 (默认)**：自动根据分辨率调整
  - 4K 以上: 4 帧/chunk
  - 1080p 以上: 8 帧/chunk
  - 720p 以上: 12 帧/chunk
  - 其他: 16 帧/chunk

- **chunk_size = 1~100**：手动指定每次处理的帧数
  - 如果显存使用率低但 GPU 使用率高，可以增大此值来加速处理
  - 如果显存不足报错，可以减小此值

## 安装

### 1. 克隆仓库

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Fluchw/comfyui_subtitle_detector.git
```

### 2. 安装依赖

```bash
cd comfyui_subtitle_detector
pip install -r requirements.txt
```

### 3. 下载模型

在 ComfyUI 根目录下执行以下命令下载模型：

```bash
# 安装 modelscope (如果没有)
pip install modelscope

# 下载 ProPainter 模型到 models/DiffuEraser/
python custom_nodes/comfyui_subtitle_detector/download_modelscope_multi.py --model Fluchw/propainter --subdir "*.pt" "*.pth" --output ./models/DiffuEraser
```

或者手动下载模型文件放入 `ComfyUI/models/DiffuEraser/` 目录：

- `ProPainter.pth` - ProPainter 主模型
- `raft-things.pth` - RAFT 光流模型
- `recurrent_flow_completion.pth` - 流补全模型

手动下载地址：[ProPainter Releases](https://github.com/sczhou/ProPainter/releases)

#### 模型文件结构

```
ComfyUI/
└── models/
    └── DiffuEraser/
        ├── ProPainter.pth                    # propainter_model 选这个
        ├── raft-things.pth                   # raft_model 选这个
        └── recurrent_flow_completion.pth     # flow_model 选这个
```

> 更多模型下载命令（包括 DiffuEraser 精修节点所需模型）请参考 [load_model.txt](load_model.txt)

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

---

## 项目结构

```
comfyui_subtitle_detector/
├── __init__.py              # ComfyUI 节点注册入口
├── nodes.py                 # SubtitleDetectorRapidOCR 节点实现
├── eraser_node.py           # SubtitleEraserProPainter 节点实现
├── rapid_ocr_engine.py      # RapidOCR 引擎封装
├── frame_renderer.py        # 检测框渲染器
├── propainter/              # ProPainter 视频修复模型（从 DiffuEraser 复制）
│   ├── inference.py         # ProPainter 推理入口
│   ├── model/               # 模型定义
│   └── RAFT/                # 光流估计模型
├── libs/                    # DiffuEraser 相关库（从 ComfyUI_DiffuEraser 复制）
│   ├── diffueraser.py       # DiffuEraser 封装
│   └── pipeline_diffueraser.py
├── sd15_repo/               # Stable Diffusion 1.5 配置文件
├── workflows/               # 示例工作流
│   └── subtitle_eraser_workflow.json
└── web/js/                  # 前端预览脚本
```

## 开发注意事项

### 1. 内存管理

**问题**：处理高分辨率长视频时容易内存溢出（RAM 和 VRAM）。

**解决方案**（在 `eraser_node.py` 中实现）：
- **分块处理**：根据分辨率动态调整 chunk_size
  - 超高清 (>1080p): 8帧/chunk
  - 1080p: 10帧/chunk
  - 720p: 16帧/chunk
  - 标清: 24帧/chunk
- **流式处理**：只加载当前 chunk 需要的帧，处理完立即释放
- **延迟创建输出 tensor**：根据 ProPainter 实际输出尺寸创建，而非预设尺寸

### 2. ProPainter 自动缩放

**问题**：ProPainter 对大分辨率视频会自动缩放到 960 宽度（见 `input video size is too large, resize to 960`）。

**解决方案**：
- 在第一个 chunk 处理后，根据实际输出尺寸 (`actual_out_w`, `actual_out_h`) 创建输出 tensor
- 处理完成后，统一 resize 回原始尺寸

### 3. 模型缓存

**实现**：使用类级别变量缓存已加载的模型
```python
class SubtitleEraserProPainter:
    _cached_model = None
    _cached_model_paths = None
```
避免重复加载模型，提升二次运行速度。

### 4. RapidOCR GPU 加速

**配置**（在 `rapid_ocr_engine.py` 中）：
```python
params = {
    "Det.engine_type": EngineType.TORCH,
    "EngineConfig.torch.use_cuda": True,
    "EngineConfig.torch.gpu_id": 0,
}
```

**注意**：即使启用 GPU，CPU 仍会有 60-70% 占用，因为图像预处理/后处理在 CPU 上进行。

### 5. 进度条

使用 ComfyUI 的 `ProgressBar` 显示处理进度：
```python
from comfy.utils import ProgressBar
pbar = ProgressBar(batch_size)
pbar.update(1)
```

### 6. 关键文件说明

| 文件 | 说明 |
|------|------|
| `nodes.py` | 字幕检测节点，调用 RapidOCR 进行 OCR |
| `eraser_node.py` | 字幕擦除节点，调用 ProPainter 进行视频修复 |
| `rapid_ocr_engine.py` | RapidOCR 封装，支持 GPU 加速和批处理 |
| `propainter/inference.py` | ProPainter 推理入口，`Propainter.forward()` 是核心方法 |

### 7. 数据流

```
ComfyUI Tensor [B,H,W,C] (0-1 float)
    ↓ 转换
PIL Image 列表
    ↓ ProPainter 处理
PIL Image 列表（可能尺寸不同）
    ↓ 转换 + Resize
ComfyUI Tensor [B,H,W,C] (0-1 float)
```

### 8. 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Out of Memory (RAM) | 长视频帧数太多 | 减小 `frame_load_cap` 或使用 `select_every_nth` 跳帧 |
| Out of Memory (VRAM) | 分辨率太高 | 自动降低 chunk_size，或手动降低输入分辨率 |
| 输出尺寸不匹配 | ProPainter 自动缩放 | 已在代码中处理，会自动 resize 回原始尺寸 |
| CPU 占用高 | RapidOCR 预处理 | 正常现象，GPU 仍在加速核心计算 |

---

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
