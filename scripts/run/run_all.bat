@echo off
echo Starting all NovelConverter services...
echo.
echo NOTE: Each service will open in a new window.
echo Close all windows to stop services.
echo.

set REPO_ROOT=%~dp0..\..

start "TTS Base :8001"    cmd /c "%REPO_ROOT%\scripts\run\run_tts_base.bat"
timeout /t 2 /nobreak > /dev/null
start "TTS Custom :8002"  cmd /c "%REPO_ROOT%\scripts\run\run_tts_custom.bat"
timeout /t 2 /nobreak > /dev/null
start "TTS Design :8003"  cmd /c "%REPO_ROOT%\scripts\run\run_tts_design.bat"
timeout /t 3 /nobreak > /dev/null
start "Orchestrator :8000" cmd /c "%REPO_ROOT%\scripts\run\run_orchestrator.bat"

echo.
echo All services starting!
echo  Orchestrator UI: http://localhost:8000
echo  TTS Base:        http://localhost:8001
echo  TTS Custom:      http://localhost:8002
echo  TTS Design:      http://localhost:8003
pause
