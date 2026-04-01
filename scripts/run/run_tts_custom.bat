@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set VENV=%REPO_ROOT%\.venv_tts_custom
set PORT=8002

cd /d "%REPO_ROOT%"
set PYTHONPATH=%REPO_ROOT%
set TTS_CUSTOM_PORT=%PORT%
set TTS_CUSTOM_USE_REAL=false

echo Starting TTS Custom Worker on port %PORT% ...
"%VENV%\Scripts\uvicorn" app.workers.tts_custom.main:app --host 0.0.0.0 --port %PORT%
