"""
Put the second fine-tune on a freshly rented GPU box, start it, watch it, and
bring the results home.

Uploads the app's reading code, the training and evaluation scripts, the page
lists and the 25 benchmark pages. Everything heavy -- the model, HWR200, the
school notebooks -- the box fetches itself from Hugging Face. Then
scripts/run_all.sh runs every stage unattended; running this again after a
dropped link or a reboot carries on where it stopped.

    python scripts/setup_server.py            # upload and start
    python scripts/setup_server.py --status   # how far it has got
    python scripts/setup_server.py --fetch    # pack and download the results
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REMOTE = os.path.join(HERE, "remote.py")

UPLOADS = [
    "scripts/run_all.sh", "scripts/pack_results.sh", "scripts/train_qwen.py",
    "scripts/eval_adapter.py", "scripts/bench_enhance.py", "scripts/fetch_hwr200.py",
    "train_pages.jsonl.gz", "holdout_pages.jsonl", "bench/pages.jsonl",
]

# A pid file, not pgrep -f: the ssh command that asks carries "run_all.sh" in
# its own command line, and pgrep would find the question instead of the job.
RUNNING = '[ -f run_all.pid ] && kill -0 "$(cat run_all.pid)" 2>/dev/null'

STATUS = r"""
cd ~/ocr 2>/dev/null || { echo 'на сервере ещё ничего нет'; exit 0; }
echo "готовые этапы: $(ls stages 2>/dev/null | grep -v max_pixels | tr '\n' ' ')"
%s && echo 'скрипт работает' || echo 'СКРИПТ НЕ ИДЁТ'
tail -n 4 logs/run_all.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
for f in logs/download.log logs/smoke.log logs/eval_base.log logs/enhance.log logs/train.log logs/eval_adapter.log; do
  [ -f "$f" ] && { echo "--- $f"; tail -c 3000 "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 2; }
done
df -h ~ | tail -n 1
""" % RUNNING


def remote(*args) -> int:
    return subprocess.call([sys.executable, REMOTE, *args], cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()

    if args.status:
        return remote("run", STATUS)
    if args.fetch:
        remote("run", "bash ~/ocr/scripts/pack_results.sh")
        os.makedirs(os.path.join(ROOT, "server_results", "run2"), exist_ok=True)
        remote("get", "ocr/results.tar.gz", "server_results/run2/results.tar.gz")
        return 0

    missing = [p for p in UPLOADS if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(f"не хватает файлов: {missing} -- сначала scripts/build_train_manifest.py")

    print("== код и списки страниц ==", flush=True)
    remote("run", "mkdir -p ~/ocr/src/ocr_project ~/ocr/scripts ~/ocr/bench/images ~/ocr/logs ~/ocr/stages")
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "ocr_project", "*.py"))):
        name = os.path.basename(path)
        remote("put", f"src/ocr_project/{name}", f"ocr/src/ocr_project/{name}")
    for rel in UPLOADS:
        remote("put", rel, f"ocr/{rel}")

    print("== проверочные страницы ==", flush=True)
    for name in sorted(os.listdir(os.path.join(ROOT, "bench", "images"))):
        remote("put", f"bench/images/{name}", f"ocr/bench/images/{name}")

    print("== запуск ==", flush=True)
    # sed: a script saved on Windows may carry CR line ends bash refuses to run.
    # setsid + nohup, or the job dies with the ssh session -- as it once did.
    return remote("run", "cd ~/ocr && sed -i 's/\\r$//' scripts/*.sh && "
                         f"if {RUNNING}; then echo 'уже идёт'; else "
                         "setsid nohup bash scripts/run_all.sh >> logs/run_all.log 2>&1 < /dev/null & "
                         "sleep 2; echo запущено; fi")


if __name__ == "__main__":
    sys.exit(main())
