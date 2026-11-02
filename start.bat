@echo off
echo ============================================================
echo   DeepFake Voice Detection System
echo ============================================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"

python app.py

if errorlevel 1 (
    echo.
    echo Application exited with error code %errorlevel%
    echo.
    pause
)
