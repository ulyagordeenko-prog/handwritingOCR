"""
Pick a PyTorch build that actually runs on this machine's GPU.

One pinned CUDA version can't serve every card: a build new enough for an
RTX 50-series (Blackwell, sm_120) may have dropped the older architectures,
and the cu121 build the project ships with predates Blackwell entirely. Both
failure modes look identical -- torch.cuda.is_available() says True and the
first real kernel launch dies with "no kernel image is available for
execution on the device".

So instead of guessing, this tries the installed build, and if it can't run
here, reinstalls from progressively older CUDA channels until one works.
Run it only if the app reports that it fell back to the CPU.
"""
import subprocess
import sys

# newest first: a modern card needs a recent build, an old card will fail
# down the list until it reaches one still compiled for its architecture
CHANNELS = [
    ("cu129", "https://download.pytorch.org/whl/cu129"),
    ("cu128", "https://download.pytorch.org/whl/cu128"),
    ("cu126", "https://download.pytorch.org/whl/cu126"),
    ("cu121", "https://download.pytorch.org/whl/cu121"),
]

PROBE = (
    "import torch;"
    "torch.zeros(8, device='cuda').sum().item();"
    "print('OK', torch.__version__, torch.cuda.get_device_name(0))"
)


def probe() -> str | None:
    """Return the working torch's description, or None if it can't run."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.startswith("OK"):
        return result.stdout.strip()
    return None


def install(index_url: str) -> bool:
    print(f"    качаю сборку (около 2.5 ГБ)...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
         "torch", "torchvision", "--index-url", index_url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("    не установилось:", result.stderr.strip().splitlines()[-1:] or "")
        return False
    return True


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch ещё не установлен — сначала запустите install.bat")
        return 1

    if not torch.cuda.is_available():
        print("Видеокарта NVIDIA не обнаружена. Приложение будет работать")
        print("на процессоре — медленнее, но правильно. Менять нечего.")
        return 0

    working = probe()
    if working:
        print(f"Всё уже в порядке: {working}")
        return 0

    try:
        card = torch.cuda.get_device_name(0)
    except Exception:
        card = "неизвестная"
    print(f"Видеокарта: {card}")
    print(f"Текущая сборка PyTorch {torch.__version__} с ней не работает.")
    print("Подбираю подходящую. Каждая попытка — большая закачка, наберитесь терпения.\n")

    for name, url in CHANNELS:
        print(f"[{name}] пробую...", flush=True)
        if not install(url):
            continue
        working = probe()
        if working:
            print(f"\nПодошла: {working}")
            print("Готово — запускайте start.bat")
            return 0
        print(f"    не подошла, пробую следующую\n")

    print("\nНи одна сборка не подошла к этой видеокарте.")
    print("Приложение будет работать на процессоре — медленнее, но правильно.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
