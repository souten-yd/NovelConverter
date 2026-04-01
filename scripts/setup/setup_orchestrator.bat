@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set VENV=%REPO_ROOT%\.venv_orchestrator

echo === Setting up orchestrator venv ===
python -m venv "%VENV%"
"%VENV%\Scripts\pip" install --upgrade pip
"%VENV%\Scripts\pip" install -r "%REPO_ROOT%\app\orchestrator\requirements.txt"

echo.
echo Orchestrator venv ready at %VENV%
echo Activate: %VENV%\Scripts\activate
