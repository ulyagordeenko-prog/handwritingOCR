"""
Put the HWR200 archives the training data refers to onto this machine.

Run on the rented box. The page list is built locally, but the images it
points at are 70 GB unpacked here -- uploading them over a home connection
would take a day. Fetching only the archives the list actually references,
straight from Hugging Face, is 8.7 GB and a few minutes on a datacentre link.

    python scripts/fetch_hwr200.py --data hwr200_pages.jsonl
"""
import argparse
import json
import os
import shutil

from huggingface_hub import hf_hub_download


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hwr200_pages.jsonl")
    ap.add_argument("--dest", default="real_data/HWR200")
    args = ap.parse_args()

    needed = sorted({json.loads(l)["zip"] for l in open(args.data, encoding="utf-8")})
    os.makedirs(args.dest, exist_ok=True)
    for name in needed:
        target = os.path.join(args.dest, name)
        if os.path.exists(target):
            print(f"  уже есть: {name}", flush=True)
            continue
        print(f"  качаю {name}...", flush=True)
        path = hf_hub_download("AntiplagiatCompany/HWR200", name, repo_type="dataset")
        # the trainer opens archives by plain path, so put a real file where it looks
        shutil.copyfile(path, target)
        print(f"  готово: {name} ({os.path.getsize(target)/1e9:.1f} ГБ)", flush=True)
    print("АРХИВЫ_НА_МЕСТЕ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
