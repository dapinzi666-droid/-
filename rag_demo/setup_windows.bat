@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "RAG_PY=python"
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "RAG_PY=py -3.11"
) else (
  py -3.12 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "RAG_PY=py -3.12"
)
%RAG_PY% -c "import sys; assert sys.version_info[:2] in ((3,11),(3,12)) and sys.maxsize > 2**32, 'Use 64-bit Python 3.11 or 3.12'"
if errorlevel 1 goto fail
if not exist ".venv\Scripts\python.exe" %RAG_PY% -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m ipykernel install --prefix .venv --name rag-demo --display-name "RAG Demo (project)"
if errorlevel 1 goto fail
".venv\Scripts\python.exe" run_demo.py --doctor
if errorlevel 1 goto fail
echo.
echo 依赖安装完成。下一步请在终端执行：
echo .\.venv\Scripts\python.exe run_demo.py --prepare
echo 然后执行 --smoke 和 --experiments。详细步骤见 docs 文件夹。
pause
exit /b 0
:fail
echo.
echo 安装未完成。请保留窗口中最后的错误信息，查看新手说明。
pause
exit /b 1
