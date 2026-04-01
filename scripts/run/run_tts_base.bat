@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set VENV=%REPO_ROOT%\.venv_tts_base
set PORT=8001

cd /d "%REPO_ROOT%"
set PYTHONPATH=%REPO_ROOT%
set TTS_BASE_PORT=%PORT%
set TTS_BASE_USE_REAL=false

echo Starting TTS Base Worker on port %PORT% ...
"%VENV%\Scripts\uvicorn" app.workers.tts_base.main:app --host 0.0.0.0 --port %PORT%
