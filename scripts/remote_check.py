"""
One command to check a page_reader.py / app.py change against the real GPU
on the rented server: git pulls the pushed commit into ~/handwritingOCR on
the server (a real clone -- see the setup note below), launches dev_check.py
detached (survives a dropped SSH link -- see scripts/remote.py's docstring
for why that matters), polls until it finishes, and prints the result.

Push first -- this pulls whatever is already on origin/main, it does not
commit or push anything itself:

    git push && python scripts/remote_check.py bench/images/101_1011.jpg
    python scripts/remote_check.py bench/images/101_1011.jpg --ref-from-bench
    python scripts/remote_check.py path.jpg --base --poll 15

Needs server_access.txt one level above the repo (see scripts/remote.py), and
~/handwritingOCR on the server to already be a clone of this repo with the
heavy, gitignored paths (real_data, bench/images, bench/pages.jsonl,
holdout_pages.jsonl, train_pages.jsonl.gz, checkpoints/qwen3vl-hwr200-full,
.venv) symlinked in from wherever they actually live -- set up once per
rental, not by this script.

Progress is tracked with a pid file, not `pgrep -f dev_check.py`: the ssh
command that asks whether it is still running carries "dev_check.py" in its
own command line, so pgrep found the question instead of the job and this
looped "still running" forever on a run that had actually already failed to
even start -- see scripts/setup_server.py's own RUNNING check for the same
bug fixed the same way earlier in this project.
"""
import argparse
import os
import subprocess
import sys
import time

# Server logs carry Cyrillic and progress-bar block characters the Windows
# console codepage can't encode -- same fix as scripts/remote.py, needed here
# too since this prints put/run output directly rather than through it.
sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REMOTE = os.path.join(HERE, "remote.py")
DEV_CHECK_LOCAL = os.path.join(HERE, "dev_check.py")
REMOTE_DIR = "handwritingOCR"


def sh(*args) -> tuple[int, str]:
    result = subprocess.run([sys.executable, REMOTE, *args], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def run_or_die(*args) -> str:
    code, out = sh(*args)
    if code != 0:
        print(out)
        sys.exit(f"команда провалилась: {args}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to a page image, relative to the repo root on the server")
    ap.add_argument("--adapter", default="checkpoints/qwen3vl-hwr200-full")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--ref-from-bench", action="store_true")
    ap.add_argument("--poll", type=int, default=20, help="seconds between progress checks")
    ap.add_argument("--timeout", type=int, default=1800, help="give up after this many seconds")
    args = ap.parse_args()

    print("git pull на сервере...", flush=True)
    print(run_or_die("run", f"cd ~/{REMOTE_DIR} && git pull && git log -1 --oneline").strip(), flush=True)
    run_or_die("put", DEV_CHECK_LOCAL, f"{REMOTE_DIR}/dev_check.py")
    run_or_die("run", f"test -f ~/{REMOTE_DIR}/dev_check.py")  # the put above really landed
    run_or_die("run", f"rm -rf ~/{REMOTE_DIR}/src/ocr_project/__pycache__; mkdir -p ~/{REMOTE_DIR}/logs")

    flags = f"--adapter {args.adapter}" if not args.base else "--base"
    if args.ref_from_bench:
        flags += " --ref-from-bench"
    # setsid wraps the whole inner sequence, not just the python launch: cd
    # and rm ran unprotected before setsid in an earlier version, and lost a
    # real race against the pty closing when the outer ssh command's bash -c
    # returned right after backgrounding this -- the kernel HUP could (and
    # once actually did, silently, exit 0 and all) kill the subshell before
    # rm even ran, disown notwithstanding, since disown only stops the SHELL
    # sending SIGHUP, not the kernel's own tty teardown. setsid up front
    # detaches from the controlling terminal before that window opens.
    inner = (f"cd ~/{REMOTE_DIR} && rm -f dev_check_out.txt logs/dev_check.log dev_check.pid && "
            f"exec env PYTHONPATH=src .venv/bin/python dev_check.py "
            f"--image '{args.image}' {flags} > logs/dev_check.log 2>&1 < /dev/null")
    # The extra `sleep 2` after backgrounding is load-bearing, not padding:
    # this ssh session's own connection closes the moment this whole command
    # returns, and closing it right after `& disown` killed the backgrounded
    # job outright on this box (systemd's per-session process/cgroup cleanup
    # on logout, most likely) even though disown should have been enough --
    # confirmed by testing without it: the pid file and log never updated at
    # all, three times, though bash itself reported exit 0 every time. A
    # couple of seconds gives the child time to actually detach before this
    # session goes away.
    cmd = f"setsid bash -c \"{inner}\" & disown; sleep 2"
    print("запускаю на сервере...", flush=True)
    run_or_die("run", cmd)

    # Three states, not two: no pid file yet (dev_check.py hasn't reached its
    # own `open("dev_check.pid", "w")` -- still starting, NOT done, even
    # though a naive `[ -f pid ] && kill -0 ... || echo DONE` reads it that
    # way and reports done after 15s on a run that has barely begun), pid
    # file present and that pid alive (running), pid file present and dead
    # (actually done).
    pid_check = f"cd ~/{REMOTE_DIR} && [ -f dev_check.pid ] && echo HAS_PID || echo NO_PID"
    alive_check = f"cd ~/{REMOTE_DIR} && kill -0 \"$(cat dev_check.pid)\" 2>/dev/null && echo ALIVE || echo DEAD"

    started = time.time()
    saw_pid = False
    while time.time() - started < args.timeout:
        time.sleep(args.poll)
        elapsed = int(time.time() - started)
        _, has_pid = sh("run", pid_check)
        if "NO_PID" in has_pid:
            if saw_pid:
                # Had a pid, now the file is gone -- dev_check.py removed it
                # itself, which it never does; treat as done rather than
                # loop forever on a state that should not occur.
                print(f"[{elapsed}s] pid-файл пропал -- считаю завершённым", flush=True)
                break
            print(f"[{elapsed}s] ещё запускается...", flush=True)
            continue
        saw_pid = True
        _, alive = sh("run", alive_check)
        if "DEAD" in alive:
            print(f"[{elapsed}s] готово, забираю результат...", flush=True)
            break
        print(f"[{elapsed}s] ещё работает...", flush=True)
    else:
        print(f"не уложилось в {args.timeout}с -- процесс может ещё работать на сервере, "
              f"проверьте вручную и заберите dev_check_out.txt позже")
        return 1

    local_out = os.path.join(ROOT, "dev_check_out.txt")
    code, get_msg = sh("get", f"{REMOTE_DIR}/dev_check_out.txt", local_out)
    if code == 0 and os.path.exists(local_out):
        with open(local_out, encoding="utf-8") as f:
            print("\n" + f.read())
    else:
        print("результат не скачался -- лог с сервера:")
        print(get_msg)
        _, log = sh("run", f"tail -c 3000 ~/{REMOTE_DIR}/logs/dev_check.log")
        print(log)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
