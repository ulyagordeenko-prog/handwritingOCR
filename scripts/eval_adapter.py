"""
Did the fine-tune help? The same pages read by the same reader the app uses,
once with the base model and once with the adapter, each page written down
as it is read.

Two sets of pages. Held-out HWR200 writers, on texts no other writer wrote,
in all three lightings (scripts/build_train_manifest.py picks them). And the
25 school notebook pages every earlier measurement used, so the new numbers
line up with the ones already on record.

Each run appends to its own file and skips pages already there, so a dropped
link or an ended rental costs one page, not the run.

    python scripts/eval_adapter.py --label base
    python scripts/eval_adapter.py --label adapter --adapter checkpoints/qwen3vl-hwr200-full
    python scripts/eval_adapter.py --compare
"""
import argparse
import json
import os
import statistics
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

# Output this much longer than the reference is not a misreading but a loop:
# the failure the first adapter showed on three pages.
RUNAWAY = 1.5
CONDITIONS = ("Сканы", "ФотоСветлое", "ФотоТемное")


def cer(reference: str, hypothesis: str) -> float:
    reference = " ".join(reference.split()).lower()
    hypothesis = " ".join(hypothesis.split()).lower()
    return Levenshtein.distance(reference, hypothesis) / len(reference) if reference else 0.0


def load_pages(holdout_path, bench_path):
    pages = []
    with open(holdout_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pages.append({"set": "holdout", "name": r["image"], "condition": r.get("condition", "?"),
                          "text": r["text"], "zip": r["zip"]})
    with open(bench_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # written on Windows; a backslash on Linux is part of the file name
            path = r["image"].replace("\\", "/")
            pages.append({"set": "bench", "name": path, "condition": "тетради",
                          "text": r["text"], "file": path})
    return pages


class Images:
    """Page images, from the HWR200 zips or from disk, decoded the way the app
    decodes them."""

    def __init__(self, zip_dir):
        self.zip_dir = zip_dir
        self.archives = {}

    def __call__(self, page):
        if "zip" in page:
            archive = self.archives.setdefault(
                page["zip"], zipfile.ZipFile(os.path.join(self.zip_dir, page["zip"])))
            data = np.frombuffer(archive.read(page["name"]), np.uint8)
        else:
            data = np.fromfile(page["file"], np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                # a line cut off when a run was stopped
                done[(r["set"], r["name"])] = r
    return done


def evaluate(args) -> int:
    import torch
    from ocr_project.page_reader import PageReader

    pages = load_pages(args.holdout, args.bench)
    out_path = f"bench/eval_{args.label}.jsonl"
    done = read_done(out_path)
    todo = [p for p in pages if (p["set"], p["name"]) not in done]
    print(f"[{args.label}] страниц {len(pages)}, уже прочитано {len(done)}", flush=True)
    if not todo:
        return 0

    # adapter_dir is always passed: left at its default, the reader would load
    # whatever adapter happens to sit at the app's path, and "base" would not be.
    reader = PageReader(quantize=True, adapter_dir=args.adapter)
    if args.adapter and not reader.adapter_loaded:
        raise SystemExit(f"адаптер не загрузился из {args.adapter}")
    images = Images(args.zip_dir)

    os.makedirs("bench", exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out:
        for i, page in enumerate(todo, 1):
            record = {k: page[k] for k in ("set", "name", "condition")}
            image = images(page)
            if image is None:
                record["error"] = "не читается"
            else:
                t0 = time.time()
                try:
                    text = "\n".join(reader.read_page(image))
                    record.update(cer=cer(page["text"], text), ref_len=len(page["text"]),
                                  out_len=len(text), seconds=round(time.time() - t0, 1), text=text)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    record["error"] = "не хватило видеопамяти"
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            shown = (f"{record['cer']:6.1%} за {record['seconds']:.0f} с" if "cer" in record
                     else record["error"])
            print(f"  [{i}/{len(todo)}] {page['set']:7s} {page['condition']:12s} {shown}", flush=True)
    return 0


def compare(_args) -> int:
    base = read_done("bench/eval_base.jsonl")
    tuned = read_done("bench/eval_adapter.jsonl")
    keys = [k for k in base if k in tuned and "cer" in base[k] and "cer" in tuned[k]]
    if not keys:
        print("нечего сравнивать")
        return 1

    def runaway(r):
        return r["out_len"] > RUNAWAY * max(1, r["ref_len"])

    groups = [("все страницы", keys),
              ("отложенные почерки", [k for k in keys if k[0] == "holdout"])]
    groups += [(f"  {c}", [k for k in keys if base[k]["condition"] == c]) for c in CONDITIONS]
    groups.append(("25 тетрадных страниц", [k for k in keys if k[0] == "bench"]))

    print(f"{'':24s} {'база':>14s} {'адаптер':>14s} {'лучше':>8s} {'зацикливаний':>14s}")
    print(f"{'':24s} {'сред.  мед.':>14s} {'сред.  мед.':>14s}")
    for label, ks in groups:
        if not ks:
            continue
        b = [base[k]["cer"] for k in ks]
        t = [tuned[k]["cer"] for k in ks]
        better = sum(1 for x, y in zip(b, t) if y < x)
        loops = f"{sum(runaway(base[k]) for k in ks)} -> {sum(runaway(tuned[k]) for k in ks)}"
        print(f"{label:24s} {statistics.mean(b):6.1%} {statistics.median(b):6.1%} "
              f"{statistics.mean(t):6.1%} {statistics.median(t):6.1%} "
              f"{better:3d}/{len(ks):<4d} {loops:>14s}")

    with open("bench/compare_texts.txt", "w", encoding="utf-8") as f:
        for k in keys:
            f.write(f"===== {k[0]} {k[1]} [{base[k]['condition']}]\n"
                    f"--- база ({base[k]['cer']:.1%})\n{base[k]['text']}\n"
                    f"--- адаптер ({tuned[k]['cer']:.1%})\n{tuned[k]['text']}\n\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=("base", "adapter"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--holdout", default="holdout_pages.jsonl")
    ap.add_argument("--bench", default="bench/pages.jsonl")
    ap.add_argument("--zip-dir", default="real_data/HWR200")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        return compare(args)
    if not args.label:
        ap.error("нужен --label или --compare")
    if args.label == "adapter" and not args.adapter:
        ap.error("для --label adapter нужен --adapter")
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
