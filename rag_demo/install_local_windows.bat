@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "RAG_TARGET_MINOR="
set "RAG_OFFLINE=0"
if /I "%~1"=="--offline" set "RAG_OFFLINE=1"
if "%RAG_OFFLINE%"=="1" if exist "local_python_minor.txt" (
  findstr /x /l /c:"3.11" /c:"3.12" "local_python_minor.txt" >nul
  if errorlevel 1 goto fail
  set /p RAG_TARGET_MINOR=<"local_python_minor.txt"
)
echo 本入口只适用于 Windows 10/11 64位 x64。首次需要联网下载依赖和模型。
echo 下载文件、模型、实验结果会保留在当前项目文件夹中。
powershell.exe -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; if([Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITECTURE','Machine') -ne 'AMD64'){throw 'This installer requires Windows x64, not ARM64 or 32-bit Windows'}"
if errorlevel 1 goto fail
call :find_python
if defined RAG_PY goto prepare

set "RAG_INSTALLER_FILE=python-3.12.10-amd64.exe"
set "RAG_INSTALLER_VERSION=3.12.10"
if "%RAG_TARGET_MINOR%"=="3.11" (
  set "RAG_INSTALLER_FILE=python-3.11.9-amd64.exe"
  set "RAG_INSTALLER_VERSION=3.11.9"
)
if not exist "software" mkdir "software"
set "RAG_INSTALLER_PATH=%CD%\software\%RAG_INSTALLER_FILE%"
if "%RAG_OFFLINE%"=="1" if not exist "%RAG_INSTALLER_PATH%" (
  echo 缺少备份中的Python安装器，无法离线恢复。请使用完整本机备份。
  goto fail
)
powershell.exe -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $p=$env:RAG_INSTALLER_PATH; if(-not(Test-Path -LiteralPath $p)){ $t=$p+'.downloading.exe'; $u='https://www.python.org/ftp/python/'+$env:RAG_INSTALLER_VERSION+'/'+$env:RAG_INSTALLER_FILE; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $t; $s=Get-AuthenticodeSignature -LiteralPath $t; if($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch '(?i)(^|,\s*)(O|CN)\s*=\s*Python Software Foundation(,|$)'){throw 'Python signature/publisher verification failed'}; Move-Item -LiteralPath $t -Destination $p }; $s=Get-AuthenticodeSignature -LiteralPath $p; if($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch '(?i)(^|,\s*)(O|CN)\s*=\s*Python Software Foundation(,|$)'){throw 'Python signature/publisher verification failed'}"
if errorlevel 1 goto fail
echo.
echo 即将打开官方Python安装窗口。请选择Install Now，完成后关闭安装窗口。
echo 若显示Modify或Repair，请不要卸载现有版本；关闭窗口并反馈截图。
start "" /wait "%RAG_INSTALLER_PATH%" InstallAllUsers=0 InstallLauncherAllUsers=0 Include_launcher=1 Include_test=0 PrependPath=0 AssociateFiles=0
if errorlevel 1 goto fail
call :find_python
if not defined RAG_PY (
  echo 没有找到匹配的Python。请保留安装窗口的提示，查看本机安装说明。
  goto fail
)

:prepare
%RAG_PY% local_setup.py %*
if errorlevel 1 goto fail
echo.
echo 本机真实自检已完成，现在打开Notebook。正式对照实验仍需要在Notebook中运行。
call "%~dp0run_local_windows.bat"
exit /b %errorlevel%

:find_python
set "RAG_PY="
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" check_local_python.py >nul 2>&1
  if not errorlevel 1 (
    set RAG_PY="%CD%\.venv\Scripts\python.exe"
    exit /b 0
  )
)
py -3.12 check_local_python.py >nul 2>&1
if not errorlevel 1 (
  set "RAG_PY=py -3.12"
  exit /b 0
)
py -3.11 check_local_python.py >nul 2>&1
if not errorlevel 1 (
  set "RAG_PY=py -3.11"
  exit /b 0
)
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  "%LocalAppData%\Programs\Python\Python312\python.exe" check_local_python.py >nul 2>&1
  if not errorlevel 1 (
    set RAG_PY="%LocalAppData%\Programs\Python\Python312\python.exe"
    exit /b 0
  )
)
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
  "%LocalAppData%\Programs\Python\Python311\python.exe" check_local_python.py >nul 2>&1
  if not errorlevel 1 (
    set RAG_PY="%LocalAppData%\Programs\Python\Python311\python.exe"
    exit /b 0
  )
)
python check_local_python.py >nul 2>&1
if not errorlevel 1 set "RAG_PY=python"
exit /b 0

:fail
echo.
echo 准备未完成，已下载的文件会保留。请截图窗口中最后的错误，不要删除整个项目。
pause
exit /b 1
