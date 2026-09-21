"""Проверка фикса зацикливания на странице 101_1011.jpg.
Запускать через check_5050\run.bat (или из корня проекта: python check_5050\test_loop_fix.py)
"""
import gc
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

# то, на что модель обучали: ~1.3 МП на страницу, у разворота две страницы
SPREAD_PIXEL_BUDGET = 2 * 1280 * 32 * 32

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("карта:", torch.cuda.get_device_name(0),
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "ГБ", flush=True)

image = cv2.imread(str(HERE / "101_1011.jpg"))
if image is None:
    sys.exit("НЕ НАЙДЕНА картинка check_5050\\101_1011.jpg")
h, w = image.shape[:2]
print(f"картинка: {w}x{h} = {w * h / 1e6:.1f} МП", flush=True)

print("Загружаю модель...", flush=True)
t = time.time()
reader = PageReader(quantize=True, adapter_dir=str(HERE / "adapter"))
print(f"модель загружена за {time.time() - t:.0f} с; адаптер подключён: {reader.adapter_loaded}", flush=True)
try:
    footprint = reader.model.get_memory_footprint() / 1024**3
    print(f"модель занимает {footprint:.1f} ГБ "
          f"(около 6 - сжатие в 4 бита работает, около 16 - НЕ работает)", flush=True)
except Exception as exc:
    print("размер модели узнать не удалось:", exc, flush=True)
print(f"выделено на карте: {torch.cuda.memory_allocated() / 1024**3:.1f} ГБ", flush=True)

mode = "полный размер"
t = time.time()
try:
    lines = reader.read_page(image)
except torch.cuda.OutOfMemoryError as exc:
    print("\nНЕ ХВАТИЛО ПАМЯТИ на полном размере:", str(exc).split(".")[0], flush=True)
    del exc
    gc.collect()
    torch.cuda.empty_cache()
    k = (SPREAD_PIXEL_BUDGET / (w * h)) ** 0.5
    small = cv2.resize(image, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    mode = f"уменьшено до {small.shape[1]}x{small.shape[0]}"
    print(f"Повторяю с уменьшенной картинкой: {mode}", flush=True)
    t = time.time()
    lines = reader.read_page(small)

print(f"\nчтение заняло {time.time() - t:.0f} с ({mode})", flush=True)
print(f"пик памяти: {torch.cuda.max_memory_allocated() / 1024**3:.1f} ГБ", flush=True)

out = HERE / "result.txt"
out.write_text("\n".join(lines) + f"\n\nрежим: {mode}\nlast_read_looped: {reader.last_read_looped}\n",
               encoding="utf-8")
print("=== ПОСЛЕДНИЕ 6 СТРОК ===")
for l in lines[-6:]:
    print(l)
print("last_read_looped:", reader.last_read_looped)
print("Полный результат:", out)
