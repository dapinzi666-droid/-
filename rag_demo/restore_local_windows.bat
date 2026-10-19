@echo off
setlocal
cd /d "%~dp0"
call "%~dp0install_local_windows.bat" --offline
exit /b %errorlevel%
