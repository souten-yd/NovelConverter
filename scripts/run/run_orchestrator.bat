@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set VENV=%REPO_ROOT%\.venv_orchestrator
set PORT=8000

if not exist "%VENV%" (
  echo ERROR: venv not found. Run scripts\setup\setup_orchestrator.bat first.
  exit /b 1
)

cd /d "%REPO_ROOT%"
set PYTHONPATH=%REPO_ROOT%
set DATABASE_URL=sqlite:///%REPO_ROOT%\data\novelconverter.db

echo Starting Orchestrator on port %PORT% ...
"%VENV%\Scripts\uvicorn" app.orchestrator.main:app --host 0.0.0.0 --port %PORT% --reload
