@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" goto fail
".venv\Scripts\python.exe" local_setup.py --verify-assets
if errorlevel 1 goto fail
".venv\Scripts\python.exe" local_backup.py
if errorlevel 1 goto fail
echo 备份已保存在项目文件夹的上一层。请保留ZIP，可再复制到U盘。
pause
exit /b 0
:fail
echo 尚未完成本机安装或备份失败。请先完成 install_local_windows.bat，并查看最后错误。
pause
exit /b 1
