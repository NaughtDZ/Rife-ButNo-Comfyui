
```markdown
# RIFE-VFI 命令行工具 (独立运行版)

基于 ComfyUI-VFI 节点封装的独立视频帧插值命令行工具，利用 RIFE 模型实现高质量慢动作视频生成。

**项目初衷**：绕过 ComfyUI 工作流中视频帧一次性加载导致的内存溢出问题，直接利用 ComfyUI 便携版环境，对视频进行高效帧插值。

## 📌 项目背景与设计思路

### 为什么需要这个工具？

在 ComfyUI 中使用 VFI 节点处理长视频（如 1000+ 帧 4K 视频）时，ComfyUI 的读取机制会将所有帧转为无压缩数据加载到内存，导致内存占用过高甚至崩溃。RIFE 虽然推理速度快，但同样受限于 ComfyUI 的视频加载机制。

这个工具通过**逐对处理**视频帧，从根本上解决了内存溢出问题：处理任意长度视频时，内存占用仅需 ~4-8GB（取决于分辨率和缩放因子）。

### 核心设计哲学

1. **借鸡下蛋**：直接复用 ComfyUI 便携版已有的完整 Python 环境和已安装的依赖（包括 `torch`、`cuda`、`opencv` 等），无需额外配置。
2. **内存友好**：逐对读取、处理、释放帧，峰值内存占用仅为单对帧的处理需求。
3. **复用节点逻辑**：核心推理代码直接取自 `ComfyUI-VFI/nodes.py`，保证与原节点行为一致。

## ✨ 特性

- 🚀 **开箱即用**：利用 ComfyUI 便携版环境，无需额外安装依赖
- 💾 **内存友好**：逐对处理视频帧，避免内存溢出（适合长视频、高分辨率视频）
- ⚡ **极速推理**：RIFE 模型以速度快著称，适合实时/准实时场景
- 🎯 **多种模型**：支持 RIFE v4.7 / v4.8 / v4.9 等多种版本
- ⚙️ **灵活配置**：支持插值倍数、缩放因子、精度等参数调整
- 🔥 **性能优化**：支持 FP16 加速，显著提升推理速度
- 🛡️ **错误恢复**：完善的错误提示，方便问题定位

## 📋 前置要求

- [ComfyUI 便携版](https://github.com/comfyanonymous/ComfyUI) (Windows 版本)
- 已安装 [ComfyUI-VFI](https://github.com/Fannovel16/ComfyUI-VFI) 节点（或包含 RIFE 实现的其他 VFI 节点）
- NVIDIA GPU（推荐 RTX 20 系列以上，4GB+ 显存即可流畅运行）

## 🚀 快速开始

### 1. 安装 ComfyUI-VFI

将 `ComfyUI-VFI` 节点放置到 ComfyUI 的 `custom_nodes` 目录：

```
ComfyUI_windows_portable/
└── ComfyUI/
    └── custom_nodes/
        └── ComfyUI-VFI/        # RIFE 节点
            ├── run_rife_cli.py # 本工具脚本
            ├── __init__.py     # 必需：让 Python 识别为包（空文件即可）
            └── ...
```

### 2. 配置脚本

打开 `run_rife_cli.py`，修改 `COMFYUI_PATH` 为你的 ComfyUI 实际路径（第 20 行左右）：

```python
COMFYUI_PATH = r"L:\Comfyui_Portable\ComfyUI_windows_portable\ComfyUI"  # 修改为你的路径
```

### 3. 运行

使用 ComfyUI 便携版的 Python 执行脚本：

```bash
# 基本用法：2倍帧插值（30fps -> 60fps）
L:\Comfyui_Portable\ComfyUI_windows_portable\python_embeded\python.exe run_rife_cli.py -i input.mp4 -o output_60fps.mp4 -n 2

# 4倍慢动作（30fps -> 120fps）
L:\Comfyui_Portable\ComfyUI_windows_portable\python_embeded\python.exe run_rife_cli.py -i input.mp4 -o output_120fps.mp4 -n 4

# 高分辨率视频（降低处理分辨率以节省显存）
L:\Comfyui_Portable\ComfyUI_windows_portable\python_embeded\python.exe run_rife_cli.py -i input.mp4 -o output.mp4 -n 2 --scale 0.5

# 使用 FP16 加速（推荐 RTX 20/30/40 系列）
L:\Comfyui_Portable\ComfyUI_windows_portable\python_embeded\python.exe run_rife_cli.py -i input.mp4 -o output.mp4 -n 2 --precision fp16
```

## 📖 命令行参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | `-i` | string | 必填 | 输入视频路径 |
| `--output` | `-o` | string | 必填 | 输出视频路径 |
| `--multiplier` | `-n` | int | 2 | 插值倍数（帧率提升倍数） |
| `--scale` | | float | 1.0 | 处理缩放因子 (0.25-1.0)，值越小显存占用越低 |
| `--model` | | string | `rife49` | RIFE 模型版本：`rife47`/`rife48`/`rife49` |
| `--precision` | | string | `fp32` | 计算精度：`fp32`/`bf16`/`fp16` |

## 💡 性能建议

### 精度选择

| 精度 | 速度 | 画质 | 适用场景 |
|------|------|------|---------|
| **FP16** | ⚡⚡⚡ 极快 | 良好 | **推荐**：日常使用，速度优先 |
| BF16 | ⚡ 较快 | 良好 | 部分 GPU 支持更好 |
| FP32 | ⚡ 一般 | 最佳 | 对画质要求极高时 |

**实测建议**：RTX 30/40 系列显卡使用 `--precision fp16` 可充分利用 Tensor Core，推理速度提升 2-3 倍。

### 分辨率与显存建议

| 视频分辨率 | 推荐 `--scale` | 建议显存 | 预期速度 |
|-----------|---------------|----------|---------|
| 1080p     | 1.0           | 4GB+     | 极快 |
| 1440p (2K)| 0.5-0.75      | 6GB+     | 快 |
| 2160p (4K)| 0.25-0.5      | 8GB+     | 中等 |
| 4320p (8K)| 0.125-0.25    | 12GB+    | 较慢 |

### 插值倍数建议

- **2x**：最常用，画质损失最小，适合日常视频补帧
- **4x**：适合慢动作效果，RIFE 表现优秀
- **8x+**：建议分段处理或降低分辨率，RIFE 在极高倍数下可能出现伪影

## ⚠️ 常见问题与故障排除

### Q1: 提示 "ComfyUI 路径无效" / `ModuleNotFoundError: No module named 'comfy'`

**解决方法**：在脚本中手动修改 `COMFYUI_PATH` 为你的 ComfyUI 实际路径。

### Q2: 提示 `ImportError: cannot import name 'RIFE_HDv3' from 'rife.model'`

**解决方法**：这说明你使用的 ComfyUI-VFI 节点版本与脚本不兼容。可以尝试：

- 更新 ComfyUI-VFI 到最新版本
- 或修改脚本中的导入语句，适配你的实际节点代码结构

### Q3: 提示 `CUDA out of memory` (显存不足)

**解决方法**：
- 降低 `--scale` 值（如 0.5 或 0.25）
- 使用 `--precision fp16` 降低显存占用
- 对于 4K 视频，建议使用 `--scale 0.5`

### Q4: 模型下载失败 / HuggingFace 连接超时

**解决方法**：
- 确保网络可以访问 HuggingFace
- 或手动下载 RIFE 模型文件（`.pkl`）到 `ComfyUI/models/rife/` 目录
- 模型下载地址：[huziwei1999/rife](https://huggingface.co/huziwei1999/rife)

### Q5: 输出视频无法播放或只有几帧

**解决方法**：
- 检查输入视频是否完整
- 尝试使用 `cv2.VideoWriter_fourcc(*'avc1')` 替换脚本中的 `mp4v`（部分播放器兼容性问题）
- 使用 VLC 或 PotPlayer 播放，这些播放器兼容性更好

### Q6: RIFE 插值出现明显伪影

**解决方法**：
- 降低插值倍数（如从 4x 降到 2x）
- 使用 `--precision fp32` 提高精度
- 尝试不同版本的 RIFE 模型（v4.7/v4.8/v4.9 各有特点）

### Q7: 处理速度比预期慢

**解决方法**：
- 使用 `--precision fp16`（RTX 20/30/40 系列速度提升显著）
- 降低 `--scale` 值
- 检查是否有其他程序占用 GPU 资源
- RIFE 本身就很快，如果还是慢可能是 CPU 瓶颈（视频解码/编码）

### Q8: 提示找不到 `__init__.py`

**解决方法**：在 `ComfyUI-VFI` 目录下创建一个空的 `__init__.py` 文件即可。这是让 Python 识别该目录为包所必需的。

## 🔧 工作原理

本工具直接复用 ComfyUI 便携版环境中已安装的依赖和 `ComfyUI-VFI` 节点的核心代码，通过以下方式实现独立运行：

1. 将 ComfyUI 路径添加到 Python 的 `sys.path`
2. 直接导入并调用 `ComfyUI-VFI` 中的 RIFE 模型类（`RIFE_HDv3`）
3. 逐对处理视频帧，避免一次性加载全部帧到内存
4. 支持 FP16/BF16/FP32 多精度推理
5. 流式写入输出视频，不保存中间帧

### RIFE 模型简介

RIFE (Real-Time Intermediate Flow Estimation) 是一种基于光流的视频帧插值算法，以其**高速推理**和**高质量插值**著称：

- **速度快**：在 RTX 3090 上可达到 30fps+ 的 4K 插值速度
- **质量高**：在多个基准测试中表现优异，伪影较少
- **版本迭代**：v4.7 ~ v4.9 持续改进，画质和速度不断提升

## 📊 与 GIMM-VFI 的对比

| 特性 | RIFE-VFI | GIMM-VFI |
|------|----------|----------|
| **速度** | ⚡⚡⚡ 极快 | ⚡ 较慢 |
| **画质** | 良好 | 更优（学术级） |
| **显存占用** | 低（4GB+） | 高（8GB+） |
| **适用场景** | 日常补帧、实时预览 | 高质量转制、影视后期 |
| **模型大小** | ~10MB | ~500MB+ |

**建议**：日常使用选 RIFE，追求极致画质选 GIMM。

## 📜 许可证

本项目基于 [ComfyUI-VFI](https://github.com/Fannovel16/ComfyUI-VFI) 和 [RIFE](https://github.com/hzwer/arXiv2020-RIFE) 开发，遵循其原始许可证（MIT/Apache 2.0）。

## 🙏 致谢

- [RIFE](https://github.com/hzwer/arXiv2020-RIFE) - 原始项目 (ICCV 2021)
- [ComfyUI-VFI](https://github.com/Fannovel16/ComfyUI-VFI) - ComfyUI 节点实现
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的节点式 AI 工具

---

**Enjoy!** 🎬 如有问题，欢迎提 Issue 或 PR。
```
