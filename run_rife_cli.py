#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RIFE 视频补帧 独立 GUI 工具 (流式版)
======================================
借用 ComfyUI 便携版自带的环境 (torch / cv2 / numpy) 和 ComfyUI-VFI 插件里的
RIFE 模型代码, 独立运行 RIFE 帧插值, 彻底绕开 ComfyUI 工作流整段读视频导致的内存爆炸。

设计要点
--------
1. 流式读取: 用 cv2 一次只解一帧, 逐对 (prev, cur) 插值后立即写盘,
   内存占用恒定, 与视频总帧数无关 (96G 内存装不下的长视频也能处理)。
2. 借鸡下蛋: 直接复用 ComfyUI 便携版 python_embeded 的 torch/cv2,
   复用 ComfyUI-VFI 的 rife 模型代码, 无需额外安装任何东西。
3. GUI: 嵌入式 Python 没有 tkinter, 因此内建一个本地 HTTP 服务, 用
   浏览器当界面 (默认自动打开), 零第三方依赖。
4. fp16 修复: 该 RIFE 代码的 warp() 在 fp16 下有 dtype 不一致的 bug,
   脚本在导入模型前注入运行时补丁修复, 不改动插件任何文件。

用法
----
用 ComfyUI 便携版的 Python 运行本脚本:

    L:\\Comfyui_Portable\\ComfyUI_windows_portable\\python_embeded\\python.exe run_rife_gui.py

也可以双击同目录的 run_rife_gui.bat 启动。
"""

import os
import sys
import json
import time
import threading
import traceback
import webbrowser
import argparse
import socket
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ==========================================================================
# 1. 路径探测: 让脚本既能放在插件里, 也能单独拷到别处
# ==========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

def _find_comfyui(start):
    """向上查找包含 comfy 包的 ComfyUI 根目录"""
    cur = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(cur, "comfy")) and os.path.exists(os.path.join(cur, "folder_paths.py")):
            return cur
        if os.path.exists(os.path.join(cur, "custom_nodes")):
            # 找含 rife 插件的自定义节点目录, 反推 ComfyUI 根
            for d in os.listdir(os.path.join(cur, "custom_nodes")):
                if os.path.isdir(os.path.join(cur, "custom_nodes", d, "rife")):
                    return cur
        # 常见: 插件目录 -> custom_nodes -> ComfyUI 根
        cur = os.path.dirname(cur)
    return None

COMFYUI_PATH = _find_comfyui(HERE) or os.environ.get("COMFYUI_PATH", "")

def _find_vfi():
    """找包含 rife 模型的 VFI 插件目录"""
    if COMFYUI_PATH and os.path.isdir(os.path.join(COMFYUI_PATH, "custom_nodes")):
        for d in os.listdir(os.path.join(COMFYUI_PATH, "custom_nodes")):
            p = os.path.join(COMFYUI_PATH, "custom_nodes", d)
            if os.path.isdir(os.path.join(p, "rife", "train_log")) or os.path.isdir(os.path.join(p, "rife", "model")):
                return p
    # 兜底: 本脚本所在目录就是插件目录
    if os.path.isdir(os.path.join(HERE, "rife")):
        return HERE
    return None

VFI_PATH = _find_vfi()

# 把 ComfyUI 根 与 VFI 插件目录 都加入 sys.path, 保证 import 可用
for p in (COMFYUI_PATH, VFI_PATH):
    if p and p not in sys.path:
        sys.path.insert(0, p)

MODEL_EXT = (".pkl", ".pt")

def list_models():
    """枚举可用的 RIFE 模型文件 (相对路径 -> 绝对路径)"""
    found = {}
    roots = []
    if VFI_PATH:
        roots.append(os.path.join(VFI_PATH, "rife", "train_log"))
        roots.append(os.path.join(VFI_PATH, "models"))
    if COMFYUI_PATH:
        roots.append(os.path.join(COMFYUI_PATH, "models", "rife"))
    for r in roots:
        if not os.path.isdir(r):
            continue
        for f in os.listdir(r):
            if f.lower().endswith(MODEL_EXT) and os.path.isfile(os.path.join(r, f)):
                found[f] = os.path.join(r, f)
    return found


# ==========================================================================
# 2. fp16 修复 (必须在导入 RIFE 模型之前注入)
# ==========================================================================
def apply_warp_fix():
    """注入 warp() 运行时补丁, 修复 fp16 下 grid_sample dtype 不一致的崩溃."""
    try:
        import torch
        from rife.model import warplayer as wp
    except ImportError:
        return  # cpu / 无 rife 包时跳过
    if getattr(wp, "_dsh_fixed", False):
        return
    _orig = wp.warp

    def warp_fixed(tenInput, tenFlow):
        # 让 flow 与输入帧 dtype 一致 (原代码在 fp16 下 flow 是 float, grid 变 float 而帧是 half)
        tenFlow = tenFlow.to(dtype=tenInput.dtype)
        k = (str(tenFlow.device), str(tenFlow.size()))
        if k not in wp.backwarp_tenGrid:
            th = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device).view(1, 1, 1, tenFlow.shape[3]).expand(
                tenFlow.shape[0], -1, tenFlow.shape[2], -1
            )
            tv = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device).view(1, 1, tenFlow.shape[2], 1).expand(
                tenFlow.shape[0], -1, -1, tenFlow.shape[3]
            )
            wp.backwarp_tenGrid[k] = torch.cat([th, tv], 1).to(tenFlow.dtype)
        grid = wp.backwarp_tenGrid[k].to(dtype=tenFlow.dtype)
        tenFlow = torch.cat(
            [
                tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
            ],
            1,
        )
        g = (grid + tenFlow).permute(0, 2, 3, 1).to(dtype=tenInput.dtype)
        return torch.nn.functional.grid_sample(
            input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True
        )

    wp.warp = warp_fixed
    wp._dsh_fixed = True


# ==========================================================================
# 3. 全局运行状态 (worker 线程写, HTTP 读)
# ==========================================================================
STATE = {
    "phase": "idle",      # idle | running | done | error
    "input": "",
    "output": "",
    "model": "",
    "n": 2,
    "use_fp16": False,
    "scale": 1.0,
    "batch": 8,
    "in_fps": 0.0,
    "out_fps": 0.0,
    "frames_in": 0,
    "frames_out": 0,
    "current": 0,          # 当前已处理原帧数
    "message": "",
    "error": "",
    "log": [],
}
LOCK = threading.Lock()
STOP_EVENT = threading.Event()
_MODEL_CACHE = {}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with LOCK:
        STATE["log"].append(line)
        if len(STATE["log"]) > 600:
            STATE["log"] = STATE["log"][-600:]
    print(line, flush=True)


# ==========================================================================
# 4. RIFE 加载与插值核心
# ==========================================================================
def get_wrapper(model_path, use_fp16):
    key = (model_path, bool(use_fp16))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    apply_warp_fix()  # 必须在 import RIFE 前调用
    import torch
    from rife.rife_comfyui_wrapper import RIFEWrapper

    log(f"加载 RIFE 模型: {model_path}  (fp16={use_fp16})")
    w = RIFEWrapper(model_path, use_fp16=bool(use_fp16 and torch.cuda.is_available()))
    _MODEL_CACHE[key] = w
    return w


def interp_pair(wrapper, I0_np, I1_np, timesteps, scale, batch):
    """对一对帧 (numpy RGB uint8 [H,W,3]) 插值, 返回 [m, H, W, 3] float32 numpy [0,1]"""
    import torch
    import numpy as np
    from torch.nn import functional as F

    model = wrapper.model
    device = wrapper.device
    dtype = torch.float16 if wrapper.use_fp16 else torch.float32

    H, W = I0_np.shape[:2]
    tmp = max(128, int(128 / scale))
    ph = ((H - 1) // tmp + 1) * tmp
    pw = ((W - 1) // tmp + 1) * tmp
    pad = (0, pw - W, 0, ph - H)

    I0 = F.pad(torch.from_numpy(I0_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0), pad).to(device, dtype)
    I1 = F.pad(torch.from_numpy(I1_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0), pad).to(device, dtype)

    res = []
    for s in range(0, len(timesteps), batch):
        chunk = timesteps[s:s + batch]
        bI0 = I0.repeat(len(chunk), 1, 1, 1)
        bI1 = I1.repeat(len(chunk), 1, 1, 1)
        with torch.inference_mode():
            out = model.inference_batch(bI0, bI1, chunk, scale=scale)
        for i in range(out.shape[0]):
            res.append(out[i, :, :H, :W].permute(1, 2, 0).cpu().float())
        del bI0, bI1, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res


# ==========================================================================
# 5. 流式视频插值主流程 (真正省内存的关键)
# ==========================================================================
def interpolate_video(input_path, output_path, model_path, n, use_fp16, scale, batch):
    import cv2
    import numpy as np
    import torch

    STOP_EVENT.clear()
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入视频不存在: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 (编解码器不受支持?): {input_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 0:
        in_fps = 30.0
        log("警告: 未能读取帧率, 回退为 30fps")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_fps = in_fps * n

    with LOCK:
        STATE.update(in_fps=in_fps, out_fps=out_fps, frames_in=frames_in, frames_out=0, current=0)

    log(f"输入: {os.path.basename(input_path)}  {W}x{H}  @{in_fps:.2f}fps  {frames_in}帧")
    log(f"目标: 补帧 ×{n}  ->  {out_fps:.2f}fps")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # 选择可用的编码器与容器
    writer, used_codec = None, None
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), out_fps, (W, H))
        if w.isOpened():
            writer, used_codec = w, fourcc
            break
    if writer is None:
        cap.release()
        raise RuntimeError("OpenCV 无可用视频编码器 (需要 avc1 或 mp4v)")

    log(f"输出编码: {used_codec}")

    wrapper = get_wrapper(model_path, use_fp16)
    n = max(1, int(n))
    timesteps = [k / n for k in range(1, n)]

    with LOCK:
        STATE["phase"] = "running"
        STATE["message"] = "处理中..."

    # 写第一帧
    ok, prev = cap.read()
    if not ok:
        cap.release(); writer.release()
        raise RuntimeError("未能读到视频第一帧")
    prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
    writer.write(prev)

    count = 1
    written = 1
    try:
        while not STOP_EVENT.is_set():
            ok, cur = cap.read()
            if not ok:
                break
            count += 1
            cur_rgb = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)

            if n > 1:
                mids = interp_pair(wrapper, prev_rgb, cur_rgb, timesteps, scale, batch)
                for m in mids:
                    m8 = (m.numpy() * 255).clip(0, 255).astype(np.uint8)
                    writer.write(cv2.cvtColor(m8, cv2.COLOR_RGB2BGR))
                    written += 1
            writer.write(cur)
            written += 1

            with LOCK:
                STATE["current"] = count
                STATE["frames_out"] = written

            prev = cur
            prev_rgb = cur_rgb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        cap.release()
        writer.release()

    if STOP_EVENT.is_set():
        with LOCK:
            STATE["phase"] = "idle"
            STATE["message"] = "已取消"
        log("已取消")
        return

    with LOCK:
        STATE["phase"] = "done"
        STATE["message"] = "完成"
        STATE["frames_out"] = written
    log(f"完成: 原 {count} 帧 -> 输出 {written} 帧 ({out_fps:.2f}fps)")
    log(f"输出文件: {output_path}")


# ==========================================================================
# 6. worker 线程封装
# ==========================================================================
def run_job(cfg):
    try:
        interpolate_video(
            cfg["input"],
            cfg["output"],
            cfg["model"],
            cfg["n"],
            cfg["use_fp16"],
            cfg["scale"],
            cfg["batch"],
        )
    except Exception as e:
        with LOCK:
            STATE["phase"] = "error"
            STATE["error"] = str(e)
            STATE["message"] = "出错"
        log("错误: " + str(e))
        log(traceback.format_exc())


# ==========================================================================
# 7. HTTP 服务 (浏览器 GUI)
# ==========================================================================
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RIFE 补帧 (独立流式版)</title>
<style>
  :root{
    --bg:#12151c; --panel:#1c222d; --panel2:#222a38; --border:#2f3a4d;
    --txt:#e7ecf3; --muted:#8b98ac; --acc:#4f8cff; --ok:#3fca7a; --err:#ff5d6c;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--txt)}
  .wrap{max-width:820px;margin:24px auto;padding:0 16px}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:12px;margin-bottom:18px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:14px}
  label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
  input,select{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--txt);
    padding:9px 10px;border-radius:7px;font-size:14px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
  .chk{display:flex;align-items:center;gap:8px;margin-top:12px}
  .chk input{width:auto}
  .chk span{color:var(--txt);font-size:13px}
  button{cursor:pointer;border:none;border-radius:7px;padding:11px 16px;font-size:14px;font-weight:600}
  #start{background:var(--acc);color:#fff;width:100%;margin-top:16px}
  #start:disabled{opacity:.5;cursor:not-allowed}
  #cancel{background:var(--panel2);color:var(--txt);border:1px solid var(--border);margin-top:8px;width:100%}
  .bar{height:10px;background:var(--panel2);border-radius:6px;overflow:hidden;margin-top:10px;border:1px solid var(--border)}
  .bar>i{display:block;height:100%;width:0;background:var(--acc);transition:width .3s}
  #prog{font-size:12px;color:var(--muted);margin-top:6px}
  #logbox{background:#0b0e14;border:1px solid var(--border);border-radius:8px;padding:10px;
    font-family:Consolas,monospace;font-size:12px;height:180px;overflow-y:auto;white-space:pre-wrap}
  .pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600}
  .pill.idle{background:var(--panel2);color:var(--muted)}
  .pill.running{background:rgba(79,140,255,.18);color:var(--acc)}
  .pill.done{background:rgba(63,202,122,.18);color:var(--ok)}
  .pill.error{background:rgba(255,93,108,.18);color:var(--err)}
  h2{font-size:14px;margin:0 0 6px;color:var(--acc)}
  .browse{display:inline-block;margin-left:6px;color:var(--acc);cursor:pointer;font-size:13px;text-decoration:underline}
  /* 文件选择 modal */
  #fmodal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);align-items:center;justify-content:center;z-index:10}
  #fmodal .sheet{background:var(--panel);border:1px solid var(--border);border-radius:10px;width:92%;max-width:640px;max-height:80vh;
    display:flex;flex-direction:column;overflow:hidden}
  #fmodal .head{padding:14px 16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
  .flist{overflow-y:auto;padding:6px}
  .fitem{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:6px;cursor:pointer}
  .fitem:hover{background:var(--panel2)}
  .fitem.dir{color:var(--acc);font-weight:600}
  #fpath{color:var(--muted);font-size:12px;padding:8px 16px;border-top:1px solid var(--border)}
  .hint{color:var(--muted);font-size:12px;margin-top:6px}
</style>
</head>
<body>
<div class="wrap">
  <h1>🎞️ RIFE 补帧 · 独立流式版</h1>
  <div class="sub">借用 ComfyUI 环境与 VFI 插件 · 逐对帧插值, 内存恒定 · 不经过 ComfyUI 工作流</div>

  <div class="card">
    <span class="pill" id="pill">空闲</span> <span id="stat_ms"></span>
  </div>

  <div class="card">
    <h2>输入 / 输出</h2>
    <label>输入视频 (绝对路径)<span class="browse" id="browseIn">📂 浏览…</span></label>
    <input id="inp" placeholder="如 D:\videos\input.mp4" spellcheck="false">
    <label>输出视频 (绝对路径)</label>
    <input id="out" placeholder="留空则自动生成" spellcheck="false">
  </div>

  <div class="card">
    <h2>插值参数</h2>
    <div class="row3">
      <div>
        <label>补帧倍数 n</label>
        <input id="n" type="number" min="2" max="32" step="1" value="2">
      </div>
      <div>
        <label>处理尺度 scale</label>
        <input id="scale" type="number" min="0.25" max="4" step="0.25" value="1.0">
      </div>
      <div>
        <label>批量 batch</label>
        <input id="batch" type="number" min="1" max="64" step="1" value="8">
      </div>
    </div>
    <label>RIFE 模型</label>
    <select id="model"></select>
    <div class="chk">
      <input id="fp16" type="checkbox">
      <span>使用 FP16 (仅 CUDA, 更快; 已自动修复该模型的 fp16 崩溃)</span>
    </div>
  </div>

  <button id="start">开始补帧</button>
  <button id="cancel" disabled>取消</button>

  <div class="card" style="margin-top:14px">
    <div class="bar"><i id="bar"></i></div>
    <div id="prog">就绪</div>
  </div>

  <div class="card">
    <h2>运行日志</h2>
    <div id="logbox"></div>
  </div>

  <p class="hint">提示: 本工具为纯本地服务, 界面就是浏览器; 关闭本页时按住 Ctrl+C 或关掉命令行即可停止。</p>
</div>

<!-- 文件选择 modal -->
<div id="fmodal">
  <div class="sheet">
    <div class="head"><b id="ftitle">选择文件</b><button id="fclose" style="background:none;color:var(--muted);font-size:20px;padding:0 6px">✕</button></div>
    <div class="flist" id="flist"></div>
    <div id="fpath"></div>
  </div>
</div>

<script>
const $=s=>document.querySelector(s);
const pill=$("#pill"), stat=$("#stat_ms"), bar=$("#bar"), prog=$("#prog"), logbox=$("#logbox");
let curMode="in", browseDir="", fp16Init=false;

function setPill(s){
  pill.className="pill "+s; pill.textContent={idle:"空闲",running:"运行中",done:"完成",error:"出错"}[s]||s;
}
async function refreshModels(){
  const m=$("#model");
  try{
    const r=await fetch("/api/models").then(x=>x.json());
    m.innerHTML="";
    for(const [name,path] of Object.entries(r.models)){
      const o=document.createElement("option"); o.value=path; o.textContent=name; m.appendChild(o);
    }
    if(m.options.length===0){ const o=document.createElement("option"); o.value=""; o.textContent="(未找到模型文件)"; m.appendChild(o);}
  }catch(e){}
}
async function poll(){
  let s;
  try{ s=await fetch("/status").then(x=>x.json()); }catch(e){ return; }
  setPill(s.phase);
  const pct = s.frames_in>0 ? Math.min(100, Math.round(s.current/s.frames_in*100)) : 0;
  bar.style.width=pct+"%";
  prog.textContent = `${s.message}  原帧 ${s.current}/${s.frames_in}  输出帧 ${s.frames_out}`+
     (s.out_fps?`  @${s.out_fps.toFixed(2)}fps`:"")+(s.error?`  |  错误: ${s.error}`:"");
  if(s.phase==="running"){ $("#start").disabled=true; $("#cancel").disabled=false; }
  else { $("#start").disabled=false; $("#cancel").disabled=true; }
  if(!fp16Init && s.cuda){ $("#fp16").checked=true; fp16Init=true; }
  // 只追加新日志
  if(s.log && s.log.length){
    const joined=s.log.map(x=>x.replace(/</g,"&lt;")).join("\n");
    if(logbox.dataset.key!==String(s.log.length)){
      logbox.dataset.key=String(s.log.length);
      logbox.textContent=joined; logbox.scrollTop=logbox.scrollHeight;
    }
  }
}
$("#start").onclick=async()=>{
  const outV=$("#out").value.trim();
  const path=$("#inp").value.trim();
  if(!path){ alert("请先选择输入视频"); return; }
  let output=outV;
  if(!output){
    const p=path.replace(/\\/g,"/"); const i=p.lastIndexOf("/");
    output=(i>=0?p.slice(0,i+1):"")+p.slice(i+1).replace(/\.[^.]+$/,"_rife"+$("#n").value+"x.mp4");
  }
  const body={input:path,output:output,model:$("#model").value,
    n:parseInt($("#n").value)||2, scale:parseFloat($("#scale").value)||1.0,
    batch:parseInt($("#batch").value)||16, use_fp16:$("#fp16").checked};
  await fetch("/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  logbox.dataset.key=""; poll();
};
$("#cancel").onclick=()=>fetch("/cancel",{method:"POST"});

// ---- 文件浏览 ----
const fmodal=$("#fmodal");
$("#browseIn").onclick=()=>{ curMode="in"; openBrowse(); };
$("#fclose").onclick=()=>fmodal.style.display="none";
function openBrowse(dir){
  fmodal.style.display="flex";
  const url="/api/browse"+(dir?"?dir="+encodeURIComponent(dir):"");
  fetch(url).then(x=>x.json()).then(d=>{
    browseDir=d.dir;
    $("#fpath").textContent=d.dir;
    $("#ftitle").textContent= curMode==="in" ? "选择输入视频" : "选择输出位置";
    const fl=$("#flist"); fl.innerHTML="";
    if(d.parent){ const p=document.createElement("div"); p.className="fitem dir";
      p.textContent="⬆ 上一级"; p.onclick=()=>openBrowse(d.parent); fl.appendChild(p); }
    for(const e of d.entries){
      const it=document.createElement("div"); it.className="fitem"+(e.is_dir?" dir":"");
      it.textContent=(e.is_dir?"📁 ":"🎬 ")+e.name;
      it.onclick=()=>{ if(e.is_dir){ openBrowse(e.path); } else if(curMode==="in"){ $("#inp").value=e.path; fmodal.style.display="none"; } };
      fl.appendChild(it);
    }
  });
}
$("#inp").addEventListener("change",()=>{ if($("#out").value.trim()===""&&$("#inp").value.trim()){ $("#out").value=$("#inp").value.replace(/\.[^.]+$/,"_rife"+($("#n").value||"2")+"x.mp4"); }});
$("#n").addEventListener("input",()=>{ if($("#inp").value.trim()){ $("#out").value=$("#inp").value.replace(/\.[^.]+$/,"_rife"+($("#n").value||"2")+"x.mp4"); }});

refreshModels(); poll(); setInterval(poll,500);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
        elif self.path.startswith("/status"):
            keys = ("phase", "input", "output", "model", "n", "use_fp16", "scale",
                    "in_fps", "out_fps", "frames_in", "frames_out", "current",
                    "message", "error", "log")
            import torch
            with LOCK:
                out = {k: STATE.get(k) for k in keys}
            out["cuda"] = torch.cuda.is_available()
            self._json(out)
        elif self.path.startswith("/api/models"):
            self._json({"models": list_models()})
        elif self.path.startswith("/api/browse"):
            try:
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query).get("dir", [""])[0]
                self._json(browse_dir(q))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path == "/start":
            try:
                n = int(self.headers.get("Content-Length", 0))
                cfg = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                self._json({"error": "bad request " + str(e)}, 400)
                return
            cfg.setdefault("n", 2); cfg.setdefault("scale", 1.0)
            cfg.setdefault("batch", 8); cfg.setdefault("use_fp16", False)
            cfg.setdefault("model", "")
            # 校验与回填
            if not cfg.get("model"):
                mods = list_models()
                if not mods:
                    self._json({"error": "未找到 RIFE 模型文件 (需要 flownet.pkl)"}, 400)
                    return
                cfg["model"] = next(iter(mods.values()))
            with LOCK:
                STATE["phase"] = "idle"; STATE["error"] = ""
            t = threading.Thread(target=run_job, args=(cfg,), daemon=True)
            t.start()
            self._json({"ok": True})
        elif self.path == "/cancel":
            STOP_EVENT.set()
            self._json({"ok": True})
        else:
            self._send(404, "not found", "text/plain")


VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv", ".wmv", ".mpg", ".mpeg", ".3gp")

def browse_dir(path):
    if not path:
        path = os.path.expanduser("~") if os.path.isdir(os.path.expanduser("~")) else os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        # 若给定的是文件, 跳到其父目录
        path = os.path.dirname(path)
    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        is_dir = os.path.isdir(full)
        if is_dir:
            entries.append({"name": name + "/", "is_dir": True, "path": full})
        elif name.lower().endswith(VIDEO_EXT):
            entries.append({"name": name, "is_dir": False, "path": full})
    parent = os.path.dirname(path)
    # 不返回空 parent 上的根自身
    return {"dir": path, "parent": parent if parent != path else None, "entries": entries}


# ==========================================================================
# 8. 入口
# ==========================================================================
def pick_free_port(pref=8899):
    for port in range(pref, pref + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


def main():
    ap = argparse.ArgumentParser(description="RIFE 视频补帧 (独立 GUI 流式版)")
    ap.add_argument("--port", type=int, default=8899, help="本地服务端口 (默认 8899)")
    ap.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    ap.add_argument("--comfyui-path", default="", help="ComfyUI 根目录")
    args = ap.parse_args()

    global COMFYUI_PATH, VFI_PATH
    if args.comfyui_path:
        COMFYUI_PATH = os.path.abspath(args.comfyui_path)
        VFI_PATH = _find_vfi()
        for p in (COMFYUI_PATH, VFI_PATH):
            if p and p not in sys.path:
                sys.path.insert(0, p)

    if COMFYUI_PATH and not os.path.isdir(os.path.join(COMFYUI_PATH, "comfy")):
        print(f"警告: 指定路径没有 comfy 包: {COMFYUI_PATH}")

    if not VFI_PATH:
        print("警告: 未找到 ComfyUI-VFI 插件目录, RIFE 模型可能无法加载。")

    port = args.port or pick_free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print("=" * 60)
    print("  RIFE 补帧工具已启动")
    print(f"  地址: {url}")
    print(f"  ComfyUI: {COMFYUI_PATH}")
    print(f"  VFI 插件: {VFI_PATH}")
    print("  关闭: 在此窗口按 Ctrl+C")
    print("=" * 60)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")
        httpd.shutdown()


if __name__ == "__main__":
    # 在入口处确保 torch 可用 (启动时即检查, 便于报错)
    import torch  # noqa: F401
    apply_warp_fix()
    main()
