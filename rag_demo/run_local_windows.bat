@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" goto not_ready
".venv\Scripts\python.exe" -c "import json; from pathlib import Path; from rag_core import check_models,sha256_file; s=json.loads(Path('local_installation.json').read_text(encoding='utf-8')); assert s.get('ready') and s.get('model_manifest_sha256')==sha256_file(Path('models/manifest.json')); check_models()"
if errorlevel 1 goto not_ready
call "%~dp0start_notebook_windows.bat"
exit /b %errorlevel%
:not_ready
echo 本机准备或模型检查尚未完成，请先运行 install_local_windows.bat。
pause
exit /b 1
