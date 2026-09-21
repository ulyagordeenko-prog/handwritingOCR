@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите install.bat в корне проекта.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" check_5050\test_loop_fix.py
echo.
pause
