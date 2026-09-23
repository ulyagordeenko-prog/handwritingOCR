"""
General-purpose one-off check against a real page, for verifying a
page_reader.py/app.py change on the server's real GPU without writing a new
throwaway script each time.

    PYTHONPATH=src .venv/bin/python dev_check.py --image bench/images/101_1011.jpg
    PYTHONPATH=src .venv/bin/python dev_check.py --image bench/images/101_1011.jpg --ref-from-bench
    PYTHONPATH=src .venv/bin/python dev_check.py --image path.jpg --base   # no adapter

Always writes UTF-8 to --out (default dev_check_out.txt) instead of stdout --
stdout through this SSH path mangles Cyrillic and it is easy to lose output
to a dropped connection or a forgotten timeout. Raises GENERATION_TIMEOUT_S
so a real slow read is not mistaken for a stuck one -- the app's own 120s
guard is tuned for a live read someone is watching, not for this.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")
# Written before anything else so a caller that launched this with
# `setsid nohup ... &` can poll for completion with a pid file (see
# scripts/run_all.sh's own `echo $$ > run_all.pid`) instead of `pgrep -f
# dev_check.py`, which matches the polling command's own command line too.
with open("dev_check.pid", "w") as _f:
    _f.write(str(os.getpid()))

import cv2
from ocr_project import page_reader as page_reader_module
from ocr_project.page_reader import PageReader

page_reader_module.GENERATION_TIMEOUT_S = 900.0


def find_reference(image_arg: str):
    """--ref-from-bench: pull the ground truth for this image out of
    bench/pages.jsonl or holdout_pages.jsonl, if it's one of ours."""
    import os
    name = image_arg.replace("\\", "/")
    for path, key in (("bench/pages.jsonl", "image"), ("holdout_pages.jsonl", "image")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r[key].replace("\\", "/").endswith(os.path.basename(name)):
                    return r["text"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--adapter", default="checkpoints/qwen3vl-hwr200-full")
    ap.add_argument("--base", action="store_true", help="no adapter, base model only")
    ap.add_argument("--ref-from-bench", action="store_true",
                     help="look up ground truth in bench/pages.jsonl or holdout_pages.jsonl")
    ap.add_argument("--out", default="dev_check_out.txt")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"не открылась: {args.image}")

    reader = PageReader(quantize=True, adapter_dir=None if args.base else args.adapter)

    t0 = time.time()
    lines = reader.read_page(image)
    elapsed = time.time() - t0
    text = "\n".join(lines)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"{args.image}\n")
        f.write(f"адаптер: {reader.adapter_loaded}, заняло {elapsed:.0f}с, "
                 f"last_read_looped={reader.last_read_looped}, строк={len(lines)}\n")
        if args.ref_from_bench:
            ref = find_reference(args.image)
            if ref is not None:
                from rapidfuzz.distance import Levenshtein
                r = " ".join(ref.split()).lower()
                h = " ".join(text.split()).lower()
                cer = Levenshtein.distance(r, h) / len(r) if r else 0.0
                f.write(f"CER: {cer:.1%}\n")
            else:
                f.write("(эталон не найден в манифестах)\n")
        f.write("\n" + text + "\n")

    print(f"готово: {elapsed:.0f}с looped={reader.last_read_looped} строк={len(lines)} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
