@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set VENV=%REPO_ROOT%\.venv_tts_design

echo === Setting up TTS Design venv ===
python -m venv "%VENV%"
"%VENV%\Scripts\pip" install --upgrade pip
"%VENV%\Scripts\pip" install -r "%REPO_ROOT%\app\workers\tts_design\requirements.txt"

echo.
echo TTS Design venv ready at %VENV%
