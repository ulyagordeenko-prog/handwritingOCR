# Обновление приложения до свежей версии с GitHub.
#
# Логика живёт здесь, а не в update.bat, по одной практической причине:
# cmd.exe дочитывает .bat по ходу выполнения, поэтому файл, который сам себя
# перезаписывает, может оборваться на середине. PowerShell читает скрипт
# целиком при запуске, так что переписать его во время работы безопасно.
# update.bat при копировании пропускается и потому никогда не меняется.

$ErrorActionPreference = "Stop"
$repo = "https://github.com/ulyagordeenko-prog/handwritingOCR/archive/refs/heads/main.zip"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Fail($message) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "  $message" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host ""
    Read-Host "Нажмите Enter, чтобы закрыть"
    exit 1
}

Write-Host "============================================"
Write-Host "  Обновление приложения"
Write-Host "============================================"
Write-Host ""

$temp = Join-Path $env:TEMP ("ocr_update_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temp | Out-Null
$zip = Join-Path $temp "update.zip"

try {
    Write-Host "[1/3] Скачиваю свежую версию..."
    Invoke-WebRequest -Uri $repo -OutFile $zip -UseBasicParsing
} catch {
    Fail "Не удалось скачать. Проверьте подключение к интернету."
}

try {
    Write-Host "[2/3] Распаковываю..."
    Expand-Archive -Path $zip -DestinationPath $temp -Force
} catch {
    Fail "Архив скачался повреждённым. Запустите обновление ещё раз."
}

$source = Get-ChildItem -Path $temp -Directory | Select-Object -First 1
if (-not $source) { Fail "В архиве не оказалось файлов приложения." }

Write-Host "[3/3] Заменяю файлы приложения..."
# Копируем только то, что пришло из архива. Скачанные модели, виртуальное
# окружение и ваши фотографии лежат в папках, которых в архиве нет, поэтому
# они остаются на месте.
Get-ChildItem -Path $source.FullName -Force | Where-Object { $_.Name -ne "update.bat" } | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $root -Recurse -Force
}

Remove-Item -Path $temp -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Файлы обновлены. Дальше — установка новых библиотек"
Write-Host "и проверка видеокарты. Это самая долгая часть."
Write-Host ""

& (Join-Path $root "install.bat")
