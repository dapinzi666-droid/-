@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "JUPYTER_CONFIG_DIR=%CD%\.cache\jupyter\config"
set "JUPYTER_RUNTIME_DIR=%CD%\.cache\jupyter\runtime"
set "JUPYTER_DATA_DIR=%CD%\.cache\jupyter\data"
set "IPYTHONDIR=%CD%\.cache\ipython"
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行 setup_windows.bat 安装依赖。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m jupyterlab task2_rag_demo.ipynb --ip=127.0.0.1
set "RAG_NOTEBOOK_STATUS=%errorlevel%"
if not "%RAG_NOTEBOOK_STATUS%"=="0" pause
exit /b %RAG_NOTEBOOK_STATUS%
