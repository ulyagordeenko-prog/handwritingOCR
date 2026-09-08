@echo off
chcp 65001 >nul
setlocal

echo ============================================
echo   Установка приложения
echo   Это займёт 10-20 минут, нужен интернет.
echo ============================================
echo.

cd /d "%~dp0"

REM uv installs into %USERPROFILE%\.local\bin, which is not on PATH until a
REM new session -- add it here so the rest of this script can use it
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>&1
if errorlevel 1 (
    echo [1/4] Устанавливаю менеджер пакетов uv...
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :fail
) else (
    echo [1/4] uv уже установлен, пропускаю.
)
echo.

echo [2/4] Устанавливаю библиотеки. Это самый долгий шаг...
uv sync
if errorlevel 1 goto :fail
echo.

echo [3/3] Проверяю, подходит ли сборка PyTorch вашей видеокарте...
uv run --no-sync python setup_torch.py
echo.

echo [4/4] Скачиваю языковую модель (529 МБ)...
if exist "models\rugpt3small\model.safetensors" (
    echo      Уже скачана, пропускаю.
) else (
    uv run python setup_language_model.py
    if errorlevel 1 (
        echo      Не удалось скачать. Приложение будет работать и без неё,
        echo      просто чуть менее точно.
    )
)
echo.

echo ============================================
echo   Готово. Запускайте файл "start.bat"
echo ============================================
echo.
pause
exit /b 0

:fail
echo.
echo ============================================
echo   Установка прервалась.
echo   Проверьте подключение к интернету
echo   и запустите этот файл ещё раз.
echo ============================================
echo.
pause
exit /b 1
