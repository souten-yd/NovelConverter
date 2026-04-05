@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 scripts\bootstrap_local_windows.py
exit /b %errorlevel%
