@echo off
REM Change directory to project root
cd /d "%~dp0.."

REM Zerve AI Setup Script for Windows
REM Quick launcher for Zerve integration setup

echo.
echo ================================================================================
echo   Zerve AI Integration Setup
echo   Cement Plant AI Optimization Platform
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and add it to your PATH
    pause
    exit /b 1
)

echo Starting Zerve AI setup...
echo.

python scripts/setup_zerve.py

if errorlevel 1 (
    echo.
    echo Setup encountered errors. Please check the output above.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   Setup Complete!
echo ================================================================================
echo.
echo Next steps:
echo   1. Read ZERVE_QUICK_START.md
echo   2. Run examples: python scripts/zerve_examples.py
echo   3. Check docs/ZERVE_INTEGRATION_GUIDE.md
echo.
pause

