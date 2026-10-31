@echo off
setlocal
title GroundUpGPT Launcher
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_app.ps1"
if %errorlevel% neq 0 (
    echo.
    echo Press any key to exit...
    pause >nul
)
exit /b 0
