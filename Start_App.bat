@echo off
title AI Support Ticket Classification Web App
echo ========================================================
echo   Starting AI Ticket Classification System...
echo ========================================================

python run.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] An error occurred while running the app.
    pause
)
