@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>&1
if errorlevel 1 (
    echo Сначала запустите файл "install.bat"
    echo.
    pause
    exit /b 1
)

echo Запускаю приложение...
echo При первом запуске скачивается модель распознавания (1.3 ГБ),
echo окно появится через несколько минут. Дальше будет быстро.
echo.

uv run --no-sync python run_app.py
if errorlevel 1 (
    echo.
    echo Приложение завершилось с ошибкой.
    echo Если в тексте выше упоминается c10.dll или похожая библиотека -
    echo установите компонент Microsoft: https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    pause
)
