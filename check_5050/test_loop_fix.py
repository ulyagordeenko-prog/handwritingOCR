"""Проверка фикса зацикливания на странице 101_1011.jpg.
Запускать из корня проекта:  .\.venv\Scripts\python.exe check_5050\test_loop_fix.py
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import cv2
import torch
from ocr_project.page_reader import PageReader

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("карта:", torch.cuda.get_device_name(0),
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "ГБ", flush=True)

image = cv2.imread(str(HERE / "101_1011.jpg"))
if image is None:
    sys.exit("НЕ НАЙДЕНА картинка check_5050\101_1011.jpg")

print("Загружаю модель (при первом запуске скачает ~17 ГБ)...", flush=True)
t = time.time()
reader = PageReader(quantize=True, adapter_dir=str(HERE / "adapter"))
print(f"модель загружена за {time.time() - t:.0f} с; адаптер подключён: {reader.adapter_loaded}", flush=True)

t = time.time()
lines = reader.read_page(image)
print(f"чтение заняло {time.time() - t:.0f} с", flush=True)

out = HERE / "result.txt"
out.write_text("\n".join(lines) + f"\n\nlast_read_looped: {reader.last_read_looped}\n", encoding="utf-8")
print("=== ПОСЛЕДНИЕ 6 СТРОК ===")
for l in lines[-6:]:
    print(l)
print("last_read_looped:", reader.last_read_looped)
print("Полный результат:", out)
