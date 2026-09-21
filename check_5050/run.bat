@echo off
chcp 65001 >nul
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%LOCALAPPDATA%\Handwriter\app\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Не найдено окружение Python: ни .venv в этой папке,
    echo ни установленного приложения Handwriter.
    echo Запустите install.bat в корне проекта.
    pause
    exit /b 1
)

echo Использую Python: %PY%
echo.
"%PY%" check_5050\test_loop_fix.py
echo.
pause
