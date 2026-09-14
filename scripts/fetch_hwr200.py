"""
Put the training and evaluation images onto the rented box.

The page lists are built locally, but the images they point at are 45 GB of
HWR200 archives plus 3 GB of school notebooks -- a day of uploading over a
home connection, minutes straight from Hugging Face on a datacentre link.
Downloads go straight to where the trainer looks (local_dir), not through
the cache and a copy, which would need the disk twice over.

    python scripts/fetch_hwr200.py
"""
import argparse
import gzip
import json
import os
import zipfile

from huggingface_hub import hf_hub_download


def rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["train_pages.jsonl.gz", "holdout_pages.jsonl"])
    ap.add_argument("--dest", default="real_data/HWR200")
    ap.add_argument("--notebooks-dest", default="real_data/school_notebooks_RU")
    args = ap.parse_args()

    needed, want_notebooks = set(), False
    for path in args.data:
        for r in rows(path):
            if "zip" in r:
                needed.add(r["zip"])
            elif r.get("source") == "notebooks":
                want_notebooks = True

    os.makedirs(args.dest, exist_ok=True)
    for name in sorted(needed):
        target = os.path.join(args.dest, name)
        if os.path.exists(target):
            print(f"  уже есть: {name}", flush=True)
            continue
        print(f"  качаю {name}...", flush=True)
        hf_hub_download("AntiplagiatCompany/HWR200", name, repo_type="dataset", local_dir=args.dest)
        print(f"  готово: {name} ({os.path.getsize(target) / 1e9:.1f} ГБ)", flush=True)

    if want_notebooks:
        images = os.path.join(args.notebooks_dest, "images")
        if os.path.isdir(images) and len(os.listdir(images)) > 1000:
            print("  тетради уже распакованы", flush=True)
        else:
            print("  качаю тетради...", flush=True)
            archive = hf_hub_download("ai-forever/school_notebooks_RU", "images.zip",
                                      repo_type="dataset", local_dir=args.notebooks_dest)
            with zipfile.ZipFile(archive) as z:
                members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
                z.extractall(args.notebooks_dest, members)
            os.remove(archive)
            print(f"  тетради: {len(os.listdir(images))} страниц", flush=True)

    print("ДАННЫЕ_НА_МЕСТЕ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
