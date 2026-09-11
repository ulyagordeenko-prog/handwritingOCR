"""
Rebuild the rented GPU box from nothing, in one command.

The first setup took about an hour of hand-typed steps and hit two failures
worth not repeating: torchvision is not optional (Qwen3-VL's processor pulls
in a video processor that requires it, so the weights load and the processor
then refuses), and the benchmark manifest is written on Windows, where a path
separator is a backslash that Linux reads as part of the filename.

    python scripts/setup_server.py            # install + upload + verify
    python scripts/setup_server.py --run 15   # ...then start the benchmark
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REMOTE = os.path.join(HERE, "remote.py")

MODULES = ["__init__.py", "app.py", "page_reader.py", "paddle_reader.py",
           "recognizer.py", "rescorer.py", "segmentation.py"]

INSTALL = r"""
set -e
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/ocr/src/ocr_project ~/ocr/scripts ~/ocr/bench/images
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv.log 2>&1
export PATH=$HOME/.local/bin:$PATH
cd ~/ocr
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch "torchvision>=0.20" \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python transformers accelerate bitsandbytes \
    opencv-python-headless scipy rapidfuzz pillow safetensors
echo УСТАНОВКА_ЗАВЕРШЕНА
"""

VERIFY = r"""
cd ~/ocr && PYTHONPATH=src .venv/bin/python -c "
import torch, torchvision, transformers, cv2, bitsandbytes
print('torch', torch.__version__, '| torchvision', torchvision.__version__)
print('видеокарта:', torch.cuda.get_device_name(0))
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from ocr_project.page_reader import PageReader
from ocr_project.paddle_reader import PaddleReader
import json, os
rows = [json.loads(l) for l in open('bench/pages.jsonl', encoding='utf-8')]
missing = [r for r in rows if not os.path.isfile(r['image'].replace(chr(92), '/'))]
print('страниц:', len(rows), '| нет на диске:', len(missing))
print('ГОТОВ' if not missing else 'НЕ ХВАТАЕТ КАРТИНОК')
"
"""


def remote(*args) -> int:
    return subprocess.call([sys.executable, REMOTE, *args], cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, metavar="PAGES",
                    help="после настройки запустить замер на N страницах")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    print("== 1. код приложения ==", flush=True)
    remote("run", "mkdir -p ~/ocr/src/ocr_project ~/ocr/scripts ~/ocr/bench/images")
    for name in MODULES:
        remote("put", f"src/ocr_project/{name}", f"ocr/src/ocr_project/{name}")
    remote("put", "scripts/bench_pages.py", "ocr/scripts/bench_pages.py")
    remote("put", "scripts/bench_models.py", "ocr/scripts/bench_models.py")
    remote("put", "scripts/train_qwen.py", "ocr/scripts/train_qwen.py")
    remote("put", "scripts/eval_adapter.py", "ocr/scripts/eval_adapter.py")
    remote("put", "scripts/fetch_hwr200.py", "ocr/scripts/fetch_hwr200.py")
    remote("put", "hwr200_pages.jsonl", "ocr/hwr200_pages.jsonl")
    remote("put", "bench/pages.jsonl", "ocr/bench/pages.jsonl")

    print("== 2. страницы для замера ==", flush=True)
    images = sorted(os.listdir(os.path.join(ROOT, "bench", "images")))
    for i, name in enumerate(images, 1):
        remote("put", f"bench/images/{name}", f"ocr/bench/images/{name}")
        print(f"   {i}/{len(images)}", end="\r", flush=True)
    print(f"   залито картинок: {len(images)}")

    if not args.skip_install:
        print("== 3. библиотеки (долго) ==", flush=True)
        # setsid+nohup or the job dies with the ssh session, as it did before
        remote("run", f"cd ~/ocr && setsid nohup bash -c {shell_quote(INSTALL)} "
                      "> ~/ocr/install.log 2>&1 < /dev/null & sleep 3; echo запущено")
        print("   идёт в фоне, смотрите: python scripts/remote.py run 'tail -3 ~/ocr/install.log'")
        return 0

    print("== 4. проверка ==", flush=True)
    remote("run", VERIFY)

    if args.run:
        print(f"== 5. замер на {args.run} страницах ==", flush=True)
        remote("run", "cd ~/ocr && setsid nohup bash -c 'cd ~/ocr && PYTHONPATH=src "
                      f".venv/bin/python -u scripts/bench_pages.py --pages {args.run} "
                      "--out bench/results.tsv' > ~/ocr/bench.log 2>&1 < /dev/null & "
                      "sleep 3; echo запущено")
    return 0


def shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\''") + "'"


if __name__ == "__main__":
    sys.exit(main())
