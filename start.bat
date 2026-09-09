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

REM Приложение запускается с --no-sync, иначе uv каждый раз возвращает
REM окружение к версии PyTorch из настроек проекта и отменяет сборку,
REM подобранную под видеокарту. Но --no-sync означает и то, что нехватку
REM библиотек никто не починит сам, поэтому сначала короткая проверка.
REM torchvision входит в проверку намеренно: оно грузится не при старте
REM приложения, а позже, когда TrOCR разбирает картинку, и рассогласованное
REM с torch падает окном про "точка входа не найдена" уже после запуска.
uv run --no-sync python -c "import cv2, torch, transformers, PIL, scipy, importlib.util; importlib.util.find_spec('torchvision') and __import__('torchvision')" >nul 2>&1
if errorlevel 1 (
    echo Не хватает библиотек, доустанавливаю. Это разовая задержка...
    uv sync
    if errorlevel 1 (
        echo.
        echo Не удалось установить. Проверьте интернет и запустите "install.bat"
        echo.
        pause
        exit /b 1
    )
    REM uv sync вернул PyTorch к версии из настроек - подбираем заново
    uv run --no-sync python setup_torch.py
    echo.
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
