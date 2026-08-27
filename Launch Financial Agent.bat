@echo off
title Financial Agent
cd /d "%~dp0"
set "PY=C:\Users\bryce\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" gui.py
if errorlevel 1 (
  echo.
  echo The app exited with an error - see the message above.
  pause
)
