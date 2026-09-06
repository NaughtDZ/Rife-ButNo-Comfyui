@echo off
chcp 65001 >nul
setlocal
set "PORTABLE=%~dp0"
cd /d "%PORTABLE%"

echo ============================================================
echo   RIFE 补帧工具 (独立流式版)
echo ============================================================
echo.
echo 正在启动, 浏览器将自动打开 http://127.0.0.1:8899/
echo 关闭本窗口即可退出。
echo.

"%PORTABLE%python_embeded\python.exe" "%PORTABLE%ComfyUI\custom_nodes\ComfyUI-VFI\run_rife_gui.py" --port 8899

echo.
echo 服务已停止。
pause
