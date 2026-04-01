@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set VENV=%REPO_ROOT%\.venv_tts_custom

echo === Setting up TTS Custom venv ===
python -m venv "%VENV%"
"%VENV%\Scripts\pip" install --upgrade pip
"%VENV%\Scripts\pip" install -r "%REPO_ROOT%\app\workers\tts_custom\requirements.txt"

echo.
echo TTS Custom venv ready at %VENV%
