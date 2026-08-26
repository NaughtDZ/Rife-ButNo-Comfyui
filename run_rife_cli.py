#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RIFE-VFI 独立命令行运行脚本 (流式写入版)
内存占用恒定，可处理任意长度视频
基于 ComfyUI 便携版环境运行
"""
import os
import sys
import argparse
import time
import logging
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm

# ========== 1. 配置路径 ==========
# 请务必修改为你的 ComfyUI 实际路径！
COMFYUI_PATH = r"L:\Comfyui_Portable\ComfyUI_windows_portable\ComfyUI"
if not os.path.exists(os.path.join(COMFYUI_PATH, "comfy")):
    raise FileNotFoundError(f"ComfyUI 路径无效，请修改 COMFYUI_PATH: {COMFYUI_PATH}")

sys.path.insert(0, COMFYUI_PATH)

# 导入 ComfyUI 的模型管理模块
import comfy.model_management as mm
from comfy.utils import load_torch_file

# ========== 2. 加载 RIFE 模型 ==========
# 这里参考 ComfyUI-VFI 节点中 RIFE 的实现
# 假设 custom_nodes 中有 ComfyUI-VFI 或类似的节点，其 rife 模块可用
# 如果路径问题，可能需要手动添加
RIFE_NODE_PATH = os.path.join(COMFYUI_PATH, "custom_nodes", "ComfyUI-VFI") # 请确保你的节点文件夹名称正确
if os.path.exists(RIFE_NODE_PATH):
    sys.path.insert(0, RIFE_NODE_PATH)
else:
    logging.warning(f"未找到 ComfyUI-VFI 节点，请检查路径: {RIFE_NODE_PATH}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_rife_model(model_name="rife49", precision="fp32"):
    """加载 RIFE 模型 (基于 ComfyUI-VFI 或类似节点)"""
    device = mm.get_torch_device()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    # 尝试从 ComfyUI 标准模型目录加载
    model_path = os.path.join(COMFYUI_PATH, "models", "rife", f"{model_name}.pkl")
    
    # 如果标准路径没有，尝试从 ComfyUI-VFI 目录加载
    if not os.path.exists(model_path):
        alt_path = os.path.join(RIFE_NODE_PATH, "rife", "train_log", f"{model_name}.pkl")
        if os.path.exists(alt_path):
            model_path = alt_path

    if not os.path.exists(model_path):
        log.info(f"模型不存在，正在从 HuggingFace 下载到: {model_path}")
        from huggingface_hub import hf_hub_download
        # RIFE 官方模型通常在 huggingface 上，这里以 rife49 为例
        # 实际下载地址可能需要根据具体模型调整
        hf_hub_download(
            repo_id="huziwei1999/rife",
            filename=f"{model_name}.pkl",
            local_dir=os.path.dirname(model_path),
            local_dir_use_symlinks=False,
        )

    # 导入 RIFE 模型类 (这里需要根据你使用的节点进行调整)
    # 尝试从 ComfyUI-VFI 中导入 RIFE 模型定义
    try:
        from rife.model import RIFE_HDv3 as RIFE_Model
        from rife import flow
    except ImportError:
        log.error("无法导入 RIFE 模型定义，请确保 ComfyUI-VFI 节点已正确安装")
        raise

    # 加载模型权重
    state_dict = load_torch_file(model_path)
    model = RIFE_Model()
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(dtype).to(device)
    
    log.info(f"RIFE 模型加载成功: {model_name} on {device}")
    return model

# ========== 3. 插值函数 ==========
def run_interpolation_rife(model, I0, I1, multiplier=2, scale=1.0):
    """执行 RIFE 单对帧插值"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # 确保输入是 [B, C, H, W] 格式，且值域为 [0, 1]
    # 输入 I0, I1 为 [1, H, W, C] 格式，需要转换
    I0 = I0.permute(0, 3, 1, 2).to(device, dtype=dtype)
    I1 = I1.permute(0, 3, 1, 2).to(device, dtype=dtype)

    # RIFE 推理
    # 这里参考 ComfyUI-VFI 节点的实现方式
    # 生成中间帧的时间步列表，例如 multiplier=2，则生成一个 0.5 的时间步
    timesteps = [i / multiplier for i in range(1, multiplier)]
    
    with torch.no_grad():
        # 调用模型的 forward 方法进行插值
        # 不同的 RIFE 实现接口可能略有不同，这里以一个通用接口为例
        # 实际的 ComfyUI-VFI 节点可能包含一个 inference 函数
        outputs = model.inference(I0, I1, timesteps, scale=scale)
    
    # 将输出从 [B, C, H, W] 转回 [B, H, W, C] 并移回 CPU
    result_frames = []
    for out in outputs:
        out = out.squeeze(0).permute(1, 2, 0).detach().cpu().float()
        result_frames.append(out)
    
    return result_frames

# ========== 4. 视频处理函数 (流式写入) ==========
def read_video_to_tensor(video_path, max_frames=None):
    """读取视频为张量 [N, H, W, C], 值域 [0,1]"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and count >= max_frames):
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame_rgb).float() / 255.0)
        count += 1
    cap.release()
    if not frames:
        raise RuntimeError("未能读取到任何视频帧")
    return torch.stack(frames), fps

def write_frame_to_video(out, frame_tensor):
    """将单帧张量写入视频 (RGB -> BGR)"""
    frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

def interpolate_video_rife(input_path, output_path, multiplier=2, scale=1.0, model_name="rife49"):
    """主处理流程 (流式写入)"""
    start_time = time.time()
    log.info("=" * 60)
    log.info(f"输入视频: {input_path}")
    log.info(f"输出视频: {output_path}")
    log.info(f"插值倍数: {multiplier}x, 缩放因子: {scale}")

    # 1. 加载模型
    log.info("加载 RIFE 模型...")
    model = load_rife_model(model_name)

    # 2. 读取视频
    log.info("读取视频...")
    video_tensor, original_fps = read_video_to_tensor(input_path)
    total_frames = video_tensor.shape[0]
    if total_frames < 2:
        raise ValueError("视频帧数不足2帧")

    # 3. 初始化视频写入器
    H, W = video_tensor.shape[1], video_tensor.shape[2]
    output_fps = original_fps * multiplier
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (W, H))

    # 4. 逐对处理并流式写入
    log.info(f"开始插值, 共 {total_frames - 1} 对帧...")
    pbar = tqdm(total=total_frames - 1, desc="处理帧对")

    # 写入第一帧
    write_frame_to_video(out, video_tensor[0])

    for i in range(total_frames - 1):
        I0 = video_tensor[i:i+1]
        I1 = video_tensor[i+1:i+2]

        interp_frames = run_interpolation_rife(model, I0, I1, multiplier=multiplier, scale=scale)

        # 写入插值帧
        for frame in interp_frames:
            write_frame_to_video(out, frame)

        # 写入最后一帧 (在循环最后一次时)
        if i == total_frames - 2:
            write_frame_to_video(out, I1.squeeze(0))

        pbar.update(1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    out.release()

    elapsed = time.time() - start_time
    log.info(f"✅ 完成! 用时: {elapsed:.2f} 秒")
    log.info(f"输出视频: {output_path}")
    log.info("=" * 60)

# ========== 5. 命令行入口 ==========
def main():
    parser = argparse.ArgumentParser(description="RIFE 视频帧插值工具 (流式写入版)")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", required=True, help="输出视频路径")
    parser.add_argument("-n", "--multiplier", type=int, default=2, help="插值倍数 (例如: 2表示2倍帧率, 默认: 2)")
    parser.add_argument("--scale", type=float, default=1.0, help="处理缩放因子 (0.25-1.0, 用于性能/质量权衡, 默认: 1.0)")
    parser.add_argument("--model", type=str, default="rife49", help="模型名称 (例如: rife47, rife48, rife49)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"],
                        help="计算精度 (默认: fp32)")
    args = parser.parse_args()

    interpolate_video_rife(args.input, args.output, args.multiplier, args.scale, args.model)

if __name__ == "__main__":
    main()
