# RIFE 补帧 独立 GUI 工具 (流式版)

不用打开 ComfyUI，也能用 ComfyUI 的环境和 RIFE 插件做视频补帧。
**界面就是浏览器**，内存恒定，长视频不再把 96G 内存吃爆。

---

## 🚀 怎么用

### 方式一：双击启动（推荐）
双击 `run_rife_gui.bat`，它会自动打开浏览器界面。

### 方式二：命令行启动
```bat
cd /d L:\Comfyui_Portable\ComfyUI_windows_portable
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-VFI\run_rife_gui.py
```
浏览器自动打开 `http://127.0.0.1:8899/`。若端口被占用会自动换端口，也可用 `--port` 指定：
```bat
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-VFI\run_rife_gui.py --port 9000
```

### GUI 里怎么填
1. **输入视频**：点 `📂 浏览…` 在浏览器里挑文件，或直接粘贴绝对路径。
2. **输出视频**：留空会自动生成 `原名_rife{n}x.mp4`（跟输入同目录）。
3. **补帧倍数 n**：填几就补几倍。如 30fps 视频填 `2` → 60fps，填 `4` → 120fps。
4. **处理尺度 scale**：默认 1.0。超高清视频显存吃紧时调小（如 0.5/0.25），会先降采样处理再插值。
5. **批量 batch**：每批插多少对子帧，越大越快但越吃显存。默认 8。
6. **使用 FP16**：默认勾选（有 CUDA 时）。比 FP32 快很多，本工具已自动修复该模型的 fp16 崩溃 bug。
7. 点 **开始补帧**，下面的进度条和日志会实时刷新；**取消** 可随时中断。

---

## 🧠 它跟 ComfyUI 直接跑 RIFE 的区别（为什么不吃内存）

| | ComfyUI 工作流 | 本工具 |
|---|---|---|
| 读视频 | 一次性把全部帧解成无压缩 16bit 张量 | 用 cv2 一次只解 **一帧** |
| 插值完 | 全部帧继续堆在显存/内存 | 逐对 (prev,cur) 插值就**立即写盘** |
| 内存 | 视频越长越高，长视频轻松破 96G | **恒定**，与视频总帧数无关 |

核心：**逐对处理 + 流式写盘**，同一时间只保留正在处理的两帧和产出的若干中间帧。

## 🔌 它借用了什么
- **依赖**：ComfyUI 便携版自带的 `python_embeded`（torch / cv2 / numpy / 你的 CUDA）。
- **模型代码**：`ComfyUI-VFI` 插件里的 `rife` 模型（RIFE HDv3）。
- **模型文件**：`ComfyUI\custom_nodes\ComfyUI-VFI\rife\train_log\flownet.pkl`（脚本会自动找到；若没有会提示，可用 `--comfyui-path` 指定 ComfyUI 根目录）。

## 📁 文件组成
- `run_rife_gui.py` —— 主体脚本（在 `ComfyUI-VFI` 插件目录里）。
- `run_rife_gui.bat` —— 一键启动（在便携版根目录）。

## ⚠️ 常见问题
- **界面没弹出**：看命令行窗口提示的地址，手动在浏览器打开；确认端口没被占用。
- **输出很小/报 OpenH264 提示**：那是 cv2 探测编码器的无害 stderr 噪音，实际输出的 H.264 (avc1) 视频有效可播。
- **CUDA out of memory**：把 `scale` 调低（0.5 / 0.25）、`batch` 调小、或关掉 FP16。
- **提示找不到模型 / 加载失败**：确认 `ComfyUI-VFI\rife\train_log\flownet.pkl` 存在；不存在可运行插件的 `rife\download_rife.py` 下载，或用 `--comfyui-path` 指定正确路径。
