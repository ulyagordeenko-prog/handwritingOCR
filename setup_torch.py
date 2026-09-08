"""
Pick a PyTorch build that actually runs on this machine's GPU.

One pinned CUDA version can't serve every card: the cu121 build the project
ships with predates the RTX 50-series (Blackwell, sm_120) entirely, while a
build new enough for those has dropped the oldest architectures. Both failure
modes look identical -- torch.cuda.is_available() says True and the first real
kernel launch dies with "no kernel image is available for execution on the
device".

So instead of guessing, this tries the installed build, and if it can't run
here, works down the CUDA channels until one does.

Two details that the first version got wrong, both invisible until someone
actually ran it on a Blackwell laptop:

- it installs with `uv pip`, not `python -m pip`. A uv-managed environment has
  no pip inside it at all, so every channel failed instantly and the script
  still reported a tidy "none of them fit".
- the launcher must run with --no-sync. A plain `uv run` re-syncs the
  environment back to the cu121 pinned in pyproject.toml, silently undoing
  this fix on the next start.

pyproject.toml is deliberately left untouched, so `git pull` keeps working;
install.bat re-checks the card after each sync and calls this script again if
the freshly synced build doesn't fit.

Run it only if the app reports that it fell back to the CPU.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# newest first: a modern card needs a recent build, an old card will fail
# down the list until it reaches one still compiled for its architecture
CHANNELS = ["cu129", "cu128", "cu126", "cu121"]

PROBE = (
    "import torch;"
    "torch.zeros(8, device='cuda').sum().item();"
    "print('OK', torch.__version__, torch.cuda.get_device_name(0))"
)


def uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "uv.exe"
    return str(fallback) if fallback.exists() else "uv"


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, cwd=ROOT)


def probe() -> str | None:
    """Return the working torch's description, or None if it can't run.

    Always in a fresh subprocess: this script must never hold torch open
    itself, or Windows refuses to replace its files during a reinstall.
    """
    result = run([uv(), "run", "--no-sync", "python", "-c", PROBE])
    if result.returncode == 0 and result.stdout.startswith("OK"):
        return result.stdout.strip()
    return None


def ask(expression: str) -> str:
    result = run([uv(), "run", "--no-sync", "python", "-c",
                  f"import torch;print({expression})"])
    return result.stdout.strip()


def install(channel: str) -> bool:
    print("    качаю сборку (около 2.5 ГБ)...", flush=True)
    result = run([uv(), "pip", "install", "--reinstall-package", "torch", "torch",
                  "--index-url", f"https://download.pytorch.org/whl/{channel}"])
    if result.returncode != 0:
        tail = (result.stderr.strip().splitlines() or [""])[-1]
        print("    не установилось:", tail)
        return False
    return True


def main() -> int:
    if not ask("int(torch.cuda.is_available())").endswith("1"):
        print("Видеокарта NVIDIA не обнаружена. Приложение будет работать")
        print("на процессоре — медленнее, но правильно. Менять нечего.")
        return 0

    working = probe()
    if working:
        print(f"Всё уже в порядке: {working}")
        return 0

    print(f"Видеокарта: {ask('torch.cuda.get_device_name(0)') or 'неизвестная'}")
    print("Установленная сборка PyTorch с ней не работает.")
    print("Подбираю подходящую. Каждая попытка — большая закачка, наберитесь терпения.\n")

    for channel in CHANNELS:
        print(f"[{channel}] пробую...", flush=True)
        if not install(channel):
            continue
        working = probe()
        if working:
            print(f"\nПодошла: {working}")
            print("Готово — запускайте start.bat")
            return 0
        print("    не подошла, пробую следующую\n")

    # Nothing fit: put back the build the project ships with, so a failed
    # search can't also break the CPU fallback that was working before.
    run([uv(), "sync", "--quiet"])
    print("\nНи одна сборка не подошла к этой видеокарте.")
    print("Приложение будет работать на процессоре — медленнее, но правильно.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
