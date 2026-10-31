@echo off
setlocal
title Stop GroundUpGPT
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_app.ps1"
if %errorlevel% neq 0 (
    echo.
    echo Press any key to exit...
    pause >nul
)
exit /b 0
