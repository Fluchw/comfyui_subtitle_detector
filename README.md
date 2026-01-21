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

### SubtitleEraserDiffuEraser

DiffuEraser 精修节点，使用扩散模型对 ProPainter 结果进行精修。

**已移除 SD1.5 完整模型依赖**，直接使用独立的 CLIP 和 VAE 模型文件。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| vae | VAE 模型文件 | sd-vae-ft-mse.safetensors |
| clip | CLIP 模型文件 | clip_l.safetensors |
| lora | 可选 LoRA 模型 (如 PCM 加速) | none |
| prompt | 提示词，描述期望的背景 | clean background, high quality |
| steps | 推理步数 | 4 |
| seed | 随机种子 | 0 |
| mask_dilation | mask 膨胀像素 | 4 |
| blended | 是否混合原图边缘 | True |

#### 工作流程

1. ProPainter 先进行基础擦除（生成 priori）
2. DiffuEraser 使用扩散模型精修细节

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

# 1. 下载 ProPainter 模型 (必需)
python custom_nodes/comfyui_subtitle_detector/download_modelscope_multi.py --model Fluchw/propainter --subdir "*.pt" "*.pth" --output ./models/DiffuEraser

# 2. 下载 DiffuEraser 模型 (精修节点需要)
modelscope download --model Kijai/DiffuEraser_comfy --local_dir ./models/DiffuEraser

# 3. 下载 VAE 模型 (精修节点需要)
modelscope download --model stabilityai/sd-vae-ft-mse diffusion_pytorch_model.safetensors --local_dir ./models/vae
# 重命名为标准格式 (Linux/Mac)
mv ./models/vae/diffusion_pytorch_model.safetensors ./models/vae/sd-vae-ft-mse.safetensors
# 或 Windows:
# ren models\vae\diffusion_pytorch_model.safetensors sd-vae-ft-mse.safetensors

# 4. CLIP 模型 (通常 ComfyUI 已有 clip_l.safetensors)
# 如果没有，可从 HuggingFace 下载 openai/clip-vit-large-patch14
```

#### 模型文件结构

```
ComfyUI/
└── models/
    ├── clip/
    │   └── clip_l.safetensors           # CLIP 文本编码器 (精修节点)
    ├── vae/
    │   └── sd-vae-ft-mse.safetensors    # VAE 模型 (精修节点)
    ├── loras/
    │   └── pcm_sd15_*.safetensors       # 可选 PCM LoRA 加速
    └── DiffuEraser/
        ├── ProPainter.pth               # ProPainter 主模型
        ├── raft-things.pth              # RAFT 光流模型
        ├── recurrent_flow_completion.pth # 流补全模型
        ├── brushnet/                    # DiffuEraser BrushNet
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors
        └── unet_main/                   # DiffuEraser UNet
            ├── config.json
            └── diffusion_pytorch_model.safetensors
```

> **注意**：DiffuEraser 精修节点已移除对完整 SD1.5 checkpoint 的依赖，只需要独立的 CLIP 和 VAE 文件。

> 更多模型下载命令请参考 [load_model.txt](load_model.txt)

## 使用示例

### 基本工作流 (ProPainter)

```
VHS_LoadVideo → SubtitleDetectorRapidOCR → SubtitleEraserProPainter → VHS_VideoCombine
```

1. 使用 `VHS_LoadVideo` 加载视频
2. 使用 `SubtitleDetectorRapidOCR` 检测字幕区域
3. 使用 `SubtitleEraserProPainter` 擦除字幕
4. 使用 `VHS_VideoCombine` 保存视频

示例工作流文件：`workflows/subtitle_eraser_workflow_vhs.json`

### 高质量工作流 (ProPainter + DiffuEraser)

```
VHS_LoadVideo → SubtitleDetectorRapidOCR → SubtitleEraserProPainter → SubtitleEraserDiffuEraser → VHS_VideoCombine
```

1. 使用 `VHS_LoadVideo` 加载视频
2. 使用 `SubtitleDetectorRapidOCR` 检测字幕区域
3. 使用 `SubtitleEraserProPainter` 生成基础擦除结果 (priori)
4. 使用 `SubtitleEraserDiffuEraser` 精修细节
5. 使用 `VHS_VideoCombine` 保存视频

示例工作流文件：`workflows/subtitle_eraser_workflow_diffueraser.json`

> **注意**：DiffuEraser 精修需要更多显存 (建议 8GB+)，但效果更好

## 参数调优指南

### 字幕检测 (SubtitleDetectorRapidOCR)

| 场景 | 参数调整 |
|------|----------|
| 检测不到字幕 | 降低 `confidence_threshold` (0.2-0.25) |
| 误检太多 (非字幕被检测) | 提高 `confidence_threshold` (0.5-0.6) |
| 小字幕检测不全 | 提高 `scale_factor` (0.9-1.0) |
| 大字幕加速处理 | 降低 `scale_factor` (0.6-0.7) |
| 字幕位置变化大 | 使用 `mask_merge_mode=union` 合并所有帧的 mask |
| 字幕位置固定 | 使用 `mask_merge_mode=per_frame` 逐帧独立 mask |

### 字幕擦除 (SubtitleEraserProPainter)

| 场景 | 参数调整 |
|------|----------|
| 字幕擦除不干净 | 增大 `mask_dilation` (6-8) |
| 擦除影响周围内容 | 减小 `mask_dilation` (2-3) |
| 追求更高质量 | 减小 `ref_stride` (5-8)，增大 `raft_iter` (25-30) |
| 追求更快速度 | 增大 `ref_stride` (15-20)，减小 `raft_iter` (10-15) |
| 快速运动场景 | 增大 `neighbor_length` (15-20) |
| 静态/慢速场景 | 减小 `neighbor_length` (5-8) |
| 显存不足 (OOM) | 减小 `chunk_size` 或设为 0 自动调整 |
| 显存充足想加速 | 增大 `chunk_size` (20-40) |

### DiffuEraser 精修 (SubtitleEraserDiffuEraser)

| 场景 | 参数调整 |
|------|----------|
| 加速推理 | 使用 `pcm_sd15_smallcfg_2step` LoRA，`steps=2` |
| 更高质量 | 使用 `pcm_sd15_smallcfg_4step` LoRA，`steps=4` |
| 边缘有字幕残留 | 增大 `mask_dilation` (6-8) |
| 边缘过渡不自然 | 尝试关闭 `blended` |
| 背景有特定内容 | 修改 `prompt`，如 "clean wall"、"blue sky" |

### 推荐配置

**高质量配置** (慢但效果好)：
- ProPainter: `ref_stride=5`, `neighbor_length=15`, `raft_iter=25`
- DiffuEraser: `lora=pcm_sd15_smallcfg_4step`, `steps=4`

**平衡配置** (默认)：
- ProPainter: `ref_stride=10`, `neighbor_length=10`, `raft_iter=20`
- DiffuEraser: `lora=pcm_sd15_smallcfg_2step`, `steps=2`

**快速配置** (追求速度)：
- 只使用 ProPainter，不使用 DiffuEraser 精修
- ProPainter: `ref_stride=15`, `neighbor_length=8`, `raft_iter=12`

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
├── eraser_node.py           # SubtitleEraserProPainter + SubtitleEraserDiffuEraser 节点实现
├── rapid_ocr_engine.py      # RapidOCR 引擎封装
├── frame_renderer.py        # 检测框渲染器
├── propainter/              # ProPainter 视频修复模型
│   ├── inference.py         # ProPainter 推理入口
│   ├── model/               # 模型定义
│   └── RAFT/                # 光流估计模型
├── libs/                    # DiffuEraser 相关库
│   ├── diffueraser.py       # DiffuEraser 封装
│   └── pipeline_diffueraser.py
├── sd15_repo/               # SD1.5 配置文件 (tokenizer, text_encoder config)
├── workflows/               # 示例工作流
│   ├── subtitle_eraser_workflow_vhs.json      # 基本工作流
│   └── subtitle_eraser_workflow_diffueraser.json  # DiffuEraser 精修工作流
└── load_model.txt           # 模型下载指南
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
