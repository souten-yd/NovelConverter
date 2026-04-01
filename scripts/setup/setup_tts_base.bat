@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set VENV=%REPO_ROOT%\.venv_tts_base

echo === Setting up TTS Base venv ===
python -m venv "%VENV%"
"%VENV%\Scripts\pip" install --upgrade pip
"%VENV%\Scripts\pip" install -r "%REPO_ROOT%\app\workers\tts_base\requirements.txt"

echo.
echo TTS Base venv ready at %VENV%
