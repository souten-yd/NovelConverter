@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set VENV=%REPO_ROOT%\.venv_tts_design
set PORT=8003

cd /d "%REPO_ROOT%"
set PYTHONPATH=%REPO_ROOT%
set TTS_DESIGN_PORT=%PORT%
set TTS_DESIGN_USE_REAL=false

echo Starting TTS Design Worker on port %PORT% ...
"%VENV%\Scripts\uvicorn" app.workers.tts_design.main:app --host 0.0.0.0 --port %PORT%
