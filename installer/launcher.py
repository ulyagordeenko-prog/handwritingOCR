"""
Handwriter.exe -- one file that installs the app and then launches it.

First run: a small window walks through the install -- the app itself from
GitHub, the package manager, the libraries, a PyTorch build that suits this
graphics card, the language model -- then adds shortcuts and an entry in
Windows' installed-apps list, and opens the app. Later runs open the app
straight away, offering to update first when GitHub has a newer version.

Everything goes into %LOCALAPPDATA%\\Handwriter, so no administrator prompt.
The models already downloaded by any earlier install live in shared caches
and are found, not fetched again.

This file is stdlib only -- it is frozen into a ~10 MB exe with PyInstaller
(installer/build.py) and must run before any of the app's own dependencies
exist.

    Handwriter.exe                 install if needed, then launch
    Handwriter.exe --uninstall     remove the app
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

APP_NAME = "Handwriter"
REPO = "ulyagordeenko-prog/handwritingOCR"
BRANCH = "main"
ZIP_URL = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"
COMMIT_API = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"

# HANDWRITER_HOME lets a test install go somewhere disposable
HOME = os.environ.get("HANDWRITER_HOME") or os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), APP_NAME)
APP_DIR = os.path.join(HOME, "app")
BIN_DIR = os.path.join(HOME, "bin")
STATE = os.path.join(HOME, "state.json")
UV = os.path.join(BIN_DIR, "uv.exe")
PYTHONW = os.path.join(APP_DIR, ".venv", "Scripts", "pythonw.exe")
PYTHON = os.path.join(APP_DIR, ".venv", "Scripts", "python.exe")
INSTALLED_EXE = os.path.join(HOME, f"{APP_NAME}.exe")
APP_LOG = os.path.join(HOME, "app.log")
UNINSTALL_KEY = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_NAME}"

NO_WINDOW = 0x08000000          # CREATE_NO_WINDOW: no console flashing up
NEW_GROUP = 0x00000200          # CREATE_NEW_PROCESS_GROUP: outlives this exe
DETACHED = 0x00000008 | NEW_GROUP   # DETACHED_PROCESS too, for the app itself

# the same imports start.bat checks, torchvision included: it loads only once
# a page is read, so a mismatch would otherwise surface minutes after launch
HEALTH_CHECK = ("import cv2, torch, transformers, PIL, scipy, importlib.util; "
                "importlib.util.find_spec('torchvision') and __import__('torchvision')")

# what an app that died on start printed when a library, not the app, is at fault
MISSING_LIBRARY = ("ModuleNotFoundError", "ImportError", "DLL load failed",
                   "WinError 126", "WinError 127")

# the files that decide what the environment holds; unchanged, an update
# leaves the libraries alone
DEPS_FILES = ("pyproject.toml", "uv.lock", ".python-version", "setup_torch.py")

READY_MARK = "HANDWRITER_READY"      # printed by the app once its window is up

PLAN_TITLES = {"install": "Установка Handwriter", "update": "Обновление Handwriter",
               "repair": "Восстановление Handwriter"}

DISK_WARN_GB = 15


def clean_env() -> dict:
    """The environment for anything this exe starts, minus what PyInstaller
    set up for the exe itself.

    A frozen exe points TCL_LIBRARY and TK_LIBRARY into its own temporary
    unpack folder, and children inherit that. The app then looked for Tk's
    init.tcl in a folder deleted the moment the installer closed, and failed
    to open its window -- seen on the first real test install."""
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("_MEI", "_PYI")) or key in ("TCL_LIBRARY", "TK_LIBRARY", "TCLLIBPATH"):
            del env[key]
    unpack = getattr(sys, "_MEIPASS", None)
    if unpack:
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                      if not p.startswith(unpack))
    return env


class Failed(Exception):
    """A step that went wrong, with a message a person can act on."""


# -- state --------------------------------------------------------------------

def load_state() -> dict:
    try:
        # -sig: a file saved from Notepad starts with a BOM, which plain utf-8
        # rejects -- and an unreadable state means a needless reinstall
        with open(STATE, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(**values):
    state = load_state()
    state.update(values)
    os.makedirs(HOME, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def installed() -> bool:
    return bool(load_state().get("installed")) and os.path.isfile(PYTHONW)


def health_known() -> bool:
    """The libraries were checked for exactly this version. Checking costs as
    long as starting the app, so it is not repeated on every launch; anything
    that breaks later shows up as an app that dies on start, which repairs."""
    state = load_state()
    return bool(state.get("sha")) and state.get("healthy") == state.get("sha")


def latest_commit(timeout=4.0) -> str | None:
    """Newest commit on GitHub, or None when offline -- the app still opens."""
    try:
        req = urllib.request.Request(COMMIT_API, headers={"User-Agent": APP_NAME})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["sha"]
    except Exception:
        return None


def deps_digest() -> str:
    h = hashlib.sha256()
    for name in DEPS_FILES:
        try:
            with open(os.path.join(APP_DIR, name), "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()


def same_bytes(a: str, b: str) -> bool:
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return hashlib.sha256(fa.read()).digest() == hashlib.sha256(fb.read()).digest()
    except OSError:
        return False


# -- the steps ----------------------------------------------------------------

class Installer:
    """Runs on a worker thread. Talks to the window only through `post`."""

    def __init__(self, post, plan="repair", shortcuts=True):
        self.post = post
        self.plan = plan
        self.shortcuts = shortcuts
        self.proc = None            # the command running now, so a close can stop it
        self.sha = None             # the version fetch_app unpacked
        self.skip_deps = False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def env(self):
        env = clean_env()
        env["PATH"] = BIN_DIR + os.pathsep + env.get("PATH", "")
        env["UV_NO_PROGRESS"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def run(self, args, what, timeout=None):
        """Run a command in the app folder, showing its output as it goes."""
        proc = self.proc = subprocess.Popen(args, cwd=APP_DIR, env=self.env(),
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            creationflags=NO_WINDOW)
        tail = []
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                tail = (tail + [line])[-15:]
                self.post("detail", line)
        proc.wait(timeout=timeout)
        if proc.returncode != 0:
            raise Failed(f"{what} не удалось.\n\n" + "\n".join(tail[-6:]))

    def download(self, url, what) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": APP_NAME})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                buf, done = io.BytesIO(), 0
                while chunk := r.read(1 << 16):
                    buf.write(chunk)
                    done += len(chunk)
                    size = f" из {total / 1e6:.0f} МБ" if total else ""
                    self.post("detail", f"{what}: {done / 1e6:.1f} МБ{size}")
                return buf.getvalue()
        except OSError as exc:
            raise Failed(f"Не получилось скачать {what}. Проверьте интернет.\n\n{exc}")

    def fetch_app(self):
        data = self.download(ZIP_URL, "приложение")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            # GitHub writes the commit into the archive's comment, so the
            # version recorded is exactly the one unpacked -- asking the API
            # separately could name a newer push, which would then never be
            # offered as an update.
            comment = z.comment.decode("ascii", "ignore").strip()
            self.sha = comment if re.fullmatch(r"[0-9a-f]{40}", comment) else latest_commit(timeout=10)
            prefix = z.namelist()[0].split("/")[0] + "/"
            os.makedirs(APP_DIR, exist_ok=True)
            # Files from the archive are written over the old ones; nothing is
            # deleted, so the environment and downloaded models stay put.
            for name in z.namelist():
                rel = name[len(prefix):]
                if not rel or name.endswith("/"):
                    continue
                target = os.path.join(APP_DIR, *rel.split("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    def fetch_uv(self):
        if os.path.isfile(UV):
            return
        data = self.download(UV_URL, "менеджер пакетов")
        os.makedirs(BIN_DIR, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if name.lower().endswith(("uv.exe", "uvx.exe")):
                    with z.open(name) as src, open(os.path.join(BIN_DIR, os.path.basename(name)), "wb") as dst:
                        shutil.copyfileobj(src, dst)
        if not os.path.isfile(UV):
            raise Failed("В скачанном архиве не нашёлся uv.exe.")

    def libraries(self):
        # An update that changed only the app's own code leaves the libraries
        # as they are: a sync would swap PyTorch back to the pinned build and
        # the next step swap it forward again -- minutes of churn for nothing.
        self.skip_deps = self.plan == "update" and load_state().get("deps") == deps_digest()
        if self.skip_deps:
            self.post("detail", "Библиотеки не изменились")
            return
        self.run([UV, "sync", "--color", "never"], "Установка библиотек")

    def torch_for_this_card(self):
        if self.skip_deps:
            return
        # uv sync puts back the PyTorch build pinned in the project; this picks
        # the one that actually runs on this card, e.g. cu129 for an RTX 50
        self.run([UV, "run", "--no-sync", "python", "setup_torch.py"], "Подбор PyTorch")
        save_state(deps=deps_digest())

    def language_model(self):
        if os.path.isfile(os.path.join(APP_DIR, "models", "rugpt3small", "model.safetensors")):
            return
        try:
            self.run([UV, "run", "--no-sync", "python", "setup_language_model.py"],
                     "Загрузка языковой модели")
        except Failed:
            # optional: the app reads without it, just slightly less well
            self.post("detail", "Языковую модель скачать не удалось — приложение будет работать и без неё")

    def place_launcher(self):
        """Keep the exe the shortcuts point at current, so fixes to it arrive
        with updates. The copy that came with the app matches the version
        installed; the running exe stands in until the repository has one."""
        source = os.path.join(APP_DIR, f"{APP_NAME}.exe")
        if not os.path.isfile(source):
            if not getattr(sys, "frozen", False):
                return
            source = sys.executable
        if same_bytes(source, INSTALLED_EXE):
            return
        if os.path.exists(INSTALLED_EXE):
            # Windows lets a running exe be renamed but not overwritten, and
            # this very exe is usually the one running the update. The old
            # copy is removed on a later start.
            os.replace(INSTALLED_EXE, os.path.join(HOME, f"{APP_NAME}.old-{int(time.time())}.exe"))
        shutil.copy2(source, INSTALLED_EXE)

    def integrate(self):
        self.place_launcher()
        # the version first: the installed-apps entry shows it
        save_state(sha=self.sha or load_state().get("sha"))
        if self.shortcuts:
            make_shortcuts()
            register_uninstall()
        save_state(installed=True, installed_at=time.strftime("%Y-%m-%d %H:%M"))

    def check(self) -> str | None:
        """None if the app's libraries import, else the end of the error."""
        if not os.path.isfile(PYTHON):
            return "Окружение приложения не создано."
        try:
            r = subprocess.run([PYTHON, "-c", HEALTH_CHECK], cwd=APP_DIR, env=self.env(),
                               capture_output=True, timeout=180, creationflags=NO_WINDOW)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        if r.returncode == 0:
            return None
        return "\n".join(r.stderr.decode("utf-8", "replace").strip().splitlines()[-4:])

    def verify(self):
        # a finished install is not proof the libraries load
        problem = self.check()
        if problem:
            raise Failed("Библиотеки приложения не загружаются.\n\n" + problem)
        save_state(healthy=load_state().get("sha"))


PLANS = {
    "install": [
        ("Скачиваю приложение", "fetch_app"),
        ("Скачиваю менеджер пакетов", "fetch_uv"),
        ("Устанавливаю библиотеки — самый долгий шаг, 10–20 минут", "libraries"),
        ("Подбираю PyTorch под вашу видеокарту", "torch_for_this_card"),
        ("Скачиваю языковую модель", "language_model"),
        ("Создаю ярлыки", "integrate"),
        ("Проверяю, что всё работает", "verify"),
    ],
    "update": [
        ("Скачиваю новую версию", "fetch_app"),
        ("Скачиваю менеджер пакетов", "fetch_uv"),
        ("Обновляю библиотеки", "libraries"),
        ("Подбираю PyTorch под вашу видеокарту", "torch_for_this_card"),
        ("Скачиваю языковую модель", "language_model"),
        ("Завершаю", "integrate"),
        ("Проверяю, что всё работает", "verify"),
    ],
    "repair": [
        ("Скачиваю менеджер пакетов", "fetch_uv"),
        ("Доустанавливаю библиотеки", "libraries"),
        ("Подбираю PyTorch под вашу видеокарту", "torch_for_this_card"),
        ("Проверяю, что всё работает", "verify"),
    ],
}


# -- Windows integration ------------------------------------------------------

def _powershell(script: str):
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                   capture_output=True, creationflags=NO_WINDOW, timeout=60)


def _shell_folder(csidl: int, fallback: str) -> str:
    """Where Windows really keeps a folder. The desktop in particular is often
    moved into OneDrive, and a shortcut put in ~\\Desktop then never shows."""
    import ctypes
    buf = ctypes.create_unicode_buffer(260)
    if ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buf) == 0 and buf.value:
        return buf.value
    return fallback


def _shortcut_paths(legacy=False):
    desktop = _shell_folder(0x0010, os.path.join(os.path.expanduser("~"), "Desktop"))
    start = _shell_folder(0x0002, os.path.join(os.environ.get("APPDATA", ""),
                                               r"Microsoft\Windows\Start Menu\Programs"))
    paths = [os.path.join(desktop, f"{APP_NAME}.lnk"), os.path.join(start, f"{APP_NAME}.lnk")]
    if legacy:
        paths.append(os.path.join(os.path.expanduser("~"), "Desktop", f"{APP_NAME}.lnk"))
    return paths


def _ps_quote(text: str) -> str:
    """A PowerShell single-quoted literal: an apostrophe in a user name would
    otherwise end the string early and the shortcut would not be made."""
    return "'" + text.replace("'", "''") + "'"


def make_shortcuts():
    for lnk in _shortcut_paths():
        os.makedirs(os.path.dirname(lnk), exist_ok=True)
        _powershell(
            f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut({_ps_quote(lnk)});"
            f"$s.TargetPath={_ps_quote(INSTALLED_EXE)};$s.WorkingDirectory={_ps_quote(HOME)};"
            f"$s.IconLocation={_ps_quote(INSTALLED_EXE + ',0')};"
            f"$s.Description='Распознавание рукописного текста';$s.Save()")


def register_uninstall():
    """An entry in Settings > Apps, per user, so the app can be removed the
    way any other is."""
    import winreg
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as k:
        for name, value in (("DisplayName", APP_NAME),
                            ("DisplayIcon", INSTALLED_EXE),
                            ("DisplayVersion", (load_state().get("sha") or "")[:7] or "1.0"),
                            ("Publisher", "Handwriter"),
                            ("InstallLocation", HOME),
                            ("UninstallString", f'"{INSTALLED_EXE}" --uninstall')):
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
        winreg.SetValueEx(k, "NoModify", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(k, "NoRepair", 0, winreg.REG_DWORD, 1)


_LAUNCHER_MUTEX = None


def claim_launcher() -> bool:
    """One Handwriter.exe at a time. A second double-click while the first is
    still installing or starting would run uv on the same folder, or open a
    second copy of the app holding a second copy of the model."""
    global _LAUNCHER_MUTEX
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    _LAUNCHER_MUTEX = kernel32.CreateMutexW(None, False, "Local\\Handwriter.launcher")
    return ctypes.get_last_error() != 183       # ERROR_ALREADY_EXISTS


def tk_windows(titles) -> list:
    """Visible Tk windows of other processes with one of these titles -- the
    app's own window, or another launcher's."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    me, found = os.getpid(), []
    buf = ctypes.create_unicode_buffer(256)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value != me and user32.IsWindowVisible(hwnd):
            user32.GetClassNameW(hwnd, buf, 256)
            if buf.value == "TkTopLevel":
                user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value in titles:
                    found.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    return found


def bring_forward(hwnd):
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)          # SW_RESTORE
    user32.SetForegroundWindow(hwnd)


def remove_old_launchers():
    for old in glob.glob(os.path.join(HOME, f"{APP_NAME}.old-*.exe")):
        try:
            os.remove(old)
        except OSError:
            pass                             # still running; next time


def uninstall() -> int:
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    set_icon(root)
    try:
        # removing files under a running app fails halfway, on the ones it holds
        if not claim_launcher() or tk_windows({APP_NAME}):
            messagebox.showwarning("Удалить Handwriter",
                                   "Handwriter сейчас открыт или устанавливается.\n\n"
                                   "Закройте его и повторите удаление.", parent=root)
            return 1
        if not messagebox.askyesno(
                "Удалить Handwriter",
                "Удалить приложение и его ярлыки?\n\n"
                "Скачанные модели распознавания лежат в общем кэше и останутся — "
                "их можно удалить отдельно.", parent=root):
            return 0
        for lnk in _shortcut_paths(legacy=True):
            try:
                os.remove(lnk)
            except OSError:
                pass
        try:
            import winreg
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
        except OSError:
            pass
        messagebox.showinfo("Handwriter", "Приложение удалено.", parent=root)
    finally:
        root.destroy()
    # This exe lives in the folder being removed, so the folder goes once it
    # has exited: retried for a minute rather than tried once after a guess.
    script = (f"$p={_ps_quote(HOME)};for($i=0;$i -lt 30;$i++){{Start-Sleep -Seconds 2;"
              "Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue;"
              "if(-not(Test-Path -LiteralPath $p)){break}}")
    subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                     cwd=os.path.dirname(HOME), creationflags=NO_WINDOW | NEW_GROUP)
    return 0


# -- the windows ----------------------------------------------------------------

def set_icon(root):
    """Tk wants an .ico file, not the exe it is inside, so the icon is bundled
    alongside (PyInstaller unpacks it to sys._MEIPASS)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    try:
        root.iconbitmap(os.path.join(base, "handwriter.ico"))
    except Exception:
        pass


def _centre(root, w, h):
    root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")


def run_with_window(plan: str, shortcuts=True, quiet=False) -> bool:
    """Show progress while a plan runs. True if it finished.

    `quiet` is for work nobody asked for -- the update that happens by itself
    at every launch. A failure there must not turn into a question: the app
    that is already installed still opens, on the version it has.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    steps = PLANS[plan]
    root = tk.Tk()
    root.title(PLAN_TITLES[plan])
    set_icon(root)
    root.resizable(False, False)
    bg, fg, muted = "#f3f3f3", "#1c1c1c", "#6b6b6b"
    root.configure(bg=bg)
    _centre(root, 560, 250)

    tk.Label(root, text=PLAN_TITLES[plan], font=("Georgia", 17), bg=bg, fg=fg).pack(anchor="w", padx=28, pady=(24, 4))
    step_var, detail_var = tk.StringVar(), tk.StringVar()
    tk.Label(root, textvariable=step_var, font=("Segoe UI", 11), bg=bg, fg=fg,
             anchor="w", justify="left", wraplength=500).pack(fill="x", padx=28, pady=(8, 10))
    overall = ttk.Progressbar(root, maximum=len(steps), length=504)
    overall.pack(padx=28)
    busy = ttk.Progressbar(root, mode="indeterminate", length=504)
    busy.pack(padx=28, pady=(6, 0))
    busy.start(12)
    tk.Label(root, textvariable=detail_var, font=("Segoe UI", 9), bg=bg, fg=muted,
             anchor="w", width=80).pack(fill="x", padx=28, pady=(10, 0))

    events: queue.Queue = queue.Queue()
    result = {"ok": False}
    worker = Installer(lambda kind, value: events.put((kind, value)), plan, shortcuts)

    def work():
        try:
            for i, (label, method) in enumerate(steps):
                events.put(("step", (i, label)))
                getattr(worker, method)()
            events.put(("done", None))
        except Failed as exc:
            events.put(("failed", str(exc)))
        except Exception as exc:          # anything unexpected still reaches the user
            events.put(("failed", f"{type(exc).__name__}: {exc}"))

    def drain():
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == "step":
                    i, label = value
                    step_var.set(f"Шаг {i + 1} из {len(steps)}. {label}")
                    overall.configure(value=i)
                elif kind == "detail":
                    detail_var.set(value[:110])
                elif kind == "done":
                    overall.configure(value=len(steps))
                    result["ok"] = True
                    root.destroy()
                    return
                elif kind == "failed":
                    busy.stop()
                    if quiet:
                        root.destroy()
                        return
                    again = messagebox.askretrycancel(PLAN_TITLES[plan], value, parent=root)
                    if again:
                        busy.start(12)
                        threading.Thread(target=work, daemon=True).start()
                    else:
                        root.destroy()
                        return
        except queue.Empty:
            pass
        root.after(80, drain)

    def on_close():
        if messagebox.askyesno(PLAN_TITLES[plan], "Прервать? Начатое продолжится со следующего запуска.",
                               parent=root):
            # a closed window must not leave a 2 GB download running behind it
            worker.stop()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=work, daemon=True).start()
    root.after(80, drain)
    root.mainloop()
    return result["ok"]


class Splash:
    """A small window shown the moment Handwriter.exe starts. Loading the
    libraries takes a good 20 seconds before the app's own window appears;
    without this, a double-click seems to do nothing and gets repeated."""

    def __init__(self, text="Открываю Handwriter…"):
        import tkinter as tk
        from tkinter import ttk
        self.closed = False
        self.root = root = tk.Tk()
        root.title(APP_NAME)
        set_icon(root)
        root.resizable(False, False)
        bg = "#f3f3f3"
        root.configure(bg=bg)
        _centre(root, 380, 130)
        tk.Label(root, text=APP_NAME, font=("Georgia", 16), bg=bg, fg="#1c1c1c").pack(anchor="w", padx=24, pady=(18, 2))
        self.text = tk.StringVar(value=text)
        tk.Label(root, textvariable=self.text, font=("Segoe UI", 10), bg=bg, fg="#6b6b6b").pack(anchor="w", padx=24)
        bar = ttk.Progressbar(root, mode="indeterminate", length=332)
        bar.pack(padx=24, pady=(12, 0))
        bar.start(12)
        root.protocol("WM_DELETE_WINDOW", self._hide)
        root.update()

    def _hide(self):
        # closing it means "don't bother": what was started carries on unseen
        self.closed = True
        self.root.withdraw()

    def run(self, job, text=None):
        """job() on a thread, the window kept alive meanwhile."""
        if text:
            self.text.set(text)
        box = {}

        def work():
            try:
                box["value"] = job()
            except Exception as exc:
                box["error"] = exc

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while t.is_alive():
            self.root.update()
            time.sleep(0.03)
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def ask(self, title, message) -> bool:
        from tkinter import messagebox
        return messagebox.askyesno(title, message, parent=self.root)

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def launch_app() -> str | None:
    """Start the app and wait for its window. None if it is running, else
    what it printed before dying.

    pythonw has no console, so an app that fails on start -- a missing
    library, a file it cannot read -- simply never appears. A test install
    did exactly that and looked like success. Its output now goes to a log,
    and an early exit is shown to the person waiting for a window."""
    log = open(APP_LOG, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([PYTHONW, "run_app.py"], cwd=APP_DIR, stdout=log, stderr=log,
                            env=clean_env(), creationflags=DETACHED, close_fds=True)
    # Until the app reports its window is up, or dies. A fixed few seconds was
    # tried first and was wrong: importing torch and transformers takes longer
    # than that, so an app that crashed at 12 s passed a 10 s watch.
    ready = False
    deadline = time.time() + 120
    while time.time() < deadline and not ready:
        time.sleep(0.5)
        if proc.poll() is not None:
            break
        try:
            with open(APP_LOG, encoding="utf-8", errors="replace") as f:
                ready = READY_MARK in f.read()
        except OSError:
            pass
    log.close()
    if proc.poll() is not None and proc.returncode != 0:
        try:
            with open(APP_LOG, encoding="utf-8", errors="replace") as f:
                return "\n".join(f.read().strip().splitlines()[-8:])
        except OSError:
            return "(журнал недоступен)"
    if ready:
        # The window exists once the marker is printed and is shown a moment
        # later. Brought forward from here, while this process still holds the
        # foreground: the app itself is not allowed to take it.
        for _ in range(30):
            found = tk_windows({APP_NAME})
            if found:
                bring_forward(found[0])
                break
            time.sleep(0.3)
    return None


def show_error(title, message):
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    set_icon(root)
    messagebox.showerror(title, message, parent=root)
    root.destroy()


def ask(title, message) -> bool:
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    set_icon(root)
    try:
        return messagebox.askyesno(title, message, parent=root)
    finally:
        root.destroy()


def enough_disk() -> bool:
    """Warned before the install, not halfway through a 6 GB one."""
    drive = os.path.splitdrive(HOME)[0] + "\\"
    try:
        free = shutil.disk_usage(drive).free / 1e9
    except OSError:
        return True
    if free >= DISK_WARN_GB:
        return True
    return ask("Установка Handwriter",
               f"На диске {drive} свободно {free:.0f} ГБ. Приложению нужно около 12 ГБ, "
               "и ещё до 20 ГБ займут модели распознавания, если их нет на этом компьютере.\n\n"
               "Всё равно установить?")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--uninstall" in argv:
        return uninstall()
    if not claim_launcher():
        # another Handwriter.exe is installing or starting: show that one
        found = tk_windows(set(PLAN_TITLES.values()) | {APP_NAME})
        if found:
            bring_forward(found[0])
        return 0
    remove_old_launchers()
    shortcuts = "--no-shortcuts" not in argv

    if installed():
        running = tk_windows({APP_NAME})
        if running:
            # already open: show it -- and never update the files under it
            bring_forward(running[0])
            return 0
        splash = Splash()
        newest = splash.run(latest_commit, "Проверяю обновления…")
        # An update is not a question any more: nobody wants to decide this at
        # every launch, and the answer is always yes. An unrecorded version
        # counts as old, or it would never be updated. Most updates change only
        # the app's own files and take seconds; the window says what is going
        # on, and closing it carries on to the app as it was.
        if newest and newest != load_state().get("sha") and not splash.closed:
            splash.close()
            run_with_window("update", shortcuts, quiet=True)
            splash = Splash()
    else:
        if not enough_disk() or not run_with_window("install", shortcuts):
            return 1
        splash = Splash()

    problem = None
    try:
        if not health_known():
            broken = splash.run(Installer(lambda *_: None).check, "Проверяю, что всё на месте…")
            if broken:
                splash.close()
                if not run_with_window("repair", shortcuts):
                    return 1
                splash = Splash()
            else:
                save_state(healthy=load_state().get("sha"))
        if "--no-launch" in argv or splash.closed:
            return 0
        problem = splash.run(launch_app)
        if problem and any(mark in problem for mark in MISSING_LIBRARY):
            # a library went missing since the last check: put it back, try once more
            save_state(healthy=None)
            splash.close()
            if not run_with_window("repair", shortcuts):
                return 1
            splash = Splash()
            problem = splash.run(launch_app)
    finally:
        splash.close()

    if problem:
        show_error("Handwriter не запустился",
                   f"Приложение закрылось сразу после запуска.\n\n{problem}\n\n"
                   f"Полный журнал: {APP_LOG}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
