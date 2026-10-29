@echo off
rem 用安装了项目依赖的 Python 启动 UI（商店版 WindowsApps\python.exe 没有依赖，
rem 会导致「开启门禁」按钮灰色）
cd /d "%~dp0"
"C:\Users\LH\AppData\Local\Programs\Python\Python311\python.exe" ui\main.py
pause
