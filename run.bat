@echo off
REM Starts the DevOps Incident Intelligence server (activates ./venv if present).
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
uvicorn app.main:app --reload
