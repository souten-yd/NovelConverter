@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

py -3.11 -c "import sys; assert sys.version_info[:2]==(3,11)" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.11 が見つかりません。`py -3.11` が使えるようにしてください。
  exit /b 1
)

py -3.11 scripts\bootstrap_local_windows.py
if errorlevel 1 (
  echo [ERROR] bootstrap に失敗しました。
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" scripts\run_local_windows.py
exit /b %errorlevel%
