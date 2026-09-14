"""
Drive the rented GPU server over SSH from here.

Windows Git Bash has no sshpass, and ssh reads the password straight from
the tty, so password auth can't be scripted through the shell -- paramiko
does it directly instead. Credentials come from a local file that stays
out of the repo and out of command lines.

    python scripts/remote.py run "nvidia-smi"
    python scripts/remote.py put local.py remote.py
    python scripts/remote.py get remote/result.txt local/result.txt
"""
import os
import sys

import paramiko

# Server logs carry progress-bar block characters and Cyrillic; the Windows
# console codepage can't encode either, and an unencodable byte would
# otherwise kill the whole command mid-stream.
sys.stdout.reconfigure(errors="replace")

CREDS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "server_access.txt")


def load_creds():
    """server_access.txt: host, port, user, password -- one per line."""
    with open(CREDS_PATH, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    host, port, user, password = lines[:4]
    return host, int(port), user, password


def connect():
    host, port, user, password = load_creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=30)
    return client


def run(command: str, quiet: bool = False) -> int:
    client = connect()
    try:
        # get_pty so long-running jobs stream output instead of buffering
        stdin, stdout, stderr = client.exec_command(command, get_pty=True, timeout=None)
        for line in iter(stdout.readline, ""):
            if not quiet:
                print(line.rstrip())
        code = stdout.channel.recv_exit_status()
        err = stderr.read().decode(errors="replace").strip()
        if err and not quiet:
            print("STDERR:", err)
        return code
    finally:
        client.close()


def put(local: str, remote: str) -> None:
    client = connect()
    try:
        sftp = client.open_sftp()
        sftp.put(local, remote)
        size = os.path.getsize(local)
        print(f"отправлено {local} -> {remote} ({size/1e6:.1f} МБ)")
        sftp.close()
    finally:
        client.close()


def get(remote: str, local: str) -> None:
    client = connect()
    try:
        sftp = client.open_sftp()
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        sftp.get(remote, local)
        print(f"получено {remote} -> {local} ({os.path.getsize(local)/1e6:.1f} МБ)")
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "run":
        sys.exit(run(sys.argv[2]))
    elif action == "put":
        put(sys.argv[2], sys.argv[3])
    elif action == "get":
        get(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown action: {action}")
