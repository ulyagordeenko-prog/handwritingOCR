@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

echo ============================================
echo   Подбор версии PyTorch под вашу видеокарту
echo ============================================
echo.
echo Запускайте это, только если приложение сообщило,
echo что работает на процессоре.
echo.
echo Может потребоваться несколько больших закачек.
echo.
pause
echo.

uv run python setup_torch.py

echo.
pause
