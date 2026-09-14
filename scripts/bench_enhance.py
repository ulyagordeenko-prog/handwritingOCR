"""
Does cleaning up a photo before reading it help? One question, one change.

Reads photographs with the base model after flattening uneven light, raising
local contrast and sharpening a touch, and compares each page with the same
page read as it is. That half comes from the base evaluation, which already
read these pages with the same model, so only half the GPU time is spent.

Photos only: the 25 notebook pages and the held-out pages shot in poor light.
A scan has no uneven light to flatten.

    python scripts/bench_enhance.py        # after eval_adapter.py --label base
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from eval_adapter import RUNAWAY, Images, cer, load_pages, read_done


def enhance(bgr):
    """Even out the light, then bring the strokes forward.

    Dividing by a heavily blurred copy of the page removes a shadow or a
    bright corner without touching the ink; CLAHE then lifts faint strokes
    locally; a light unsharp mask restores edges the blur of a phone lens
    softened. The blur is done on a tenth-size copy -- at full size a kernel
    that wide takes seconds per page."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    h, w = lightness.shape
    small = cv2.resize(lightness, (max(1, w // 10), max(1, h // 10)), interpolation=cv2.INTER_AREA)
    background = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(h, w) / 300),
                            (w, h), interpolation=cv2.INTER_LINEAR)
    flat = cv2.divide(lightness, np.maximum(background, 1), scale=255)
    lifted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)
    sharp = cv2.addWeighted(lifted, 1.5, cv2.GaussianBlur(lifted, (0, 0), 1.2), -0.5, 0)
    return cv2.cvtColor(cv2.merge((sharp, a, b)), cv2.COLOR_LAB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="holdout_pages.jsonl")
    ap.add_argument("--bench", default="bench/pages.jsonl")
    ap.add_argument("--zip-dir", default="real_data/HWR200")
    ap.add_argument("--save-example", default="bench/enhance_example.jpg")
    args = ap.parse_args()

    base = read_done("bench/eval_base.jsonl")
    pages = [p for p in load_pages(args.holdout, args.bench)
             if (p["set"] == "bench" or p["condition"] == "ФотоТемное")
             and "cer" in base.get((p["set"], p["name"]), {})]
    out_path = "bench/enhance.jsonl"
    done = read_done(out_path)
    todo = [p for p in pages if (p["set"], p["name"]) not in done]
    print(f"страниц {len(pages)}, уже прочитано {len(done)}", flush=True)

    if todo:
        import torch
        from ocr_project.page_reader import PageReader

        reader = PageReader(quantize=True, adapter_dir=None)
        images = Images(args.zip_dir)
        with open(out_path, "a", encoding="utf-8") as out:
            for i, page in enumerate(todo, 1):
                image = images(page)
                if image is None:
                    continue
                better_looking = enhance(image)
                if i == 1 and args.save_example:
                    cv2.imwrite(args.save_example, np.hstack([image, better_looking]))
                t0 = time.time()
                record = {k: page[k] for k in ("set", "name", "condition")}
                try:
                    text = "\n".join(reader.read_page(better_looking))
                    record.update(cer=cer(page["text"], text), ref_len=len(page["text"]),
                                  out_len=len(text), seconds=round(time.time() - t0, 1), text=text)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    record["error"] = "не хватило видеопамяти"
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                was = base[(page["set"], page["name"])]["cer"]
                now = f"{record['cer']:6.1%}" if "cer" in record else record["error"]
                print(f"  [{i}/{len(todo)}] {page['condition']:12s} как есть {was:6.1%} -> "
                      f"улучшено {now}", flush=True)

    enhanced = read_done(out_path)
    keys = [k for k in enhanced if "cer" in enhanced[k]]
    for label, ks in (("все фото", keys),
                      ("тетради", [k for k in keys if k[0] == "bench"]),
                      ("тёмные фото HWR200", [k for k in keys if k[0] == "holdout"])):
        if not ks:
            continue
        b = [base[k]["cer"] for k in ks]
        e = [enhanced[k]["cer"] for k in ks]
        loops_b = sum(base[k]["out_len"] > RUNAWAY * max(1, base[k]["ref_len"]) for k in ks)
        loops_e = sum(enhanced[k]["out_len"] > RUNAWAY * max(1, enhanced[k]["ref_len"]) for k in ks)
        print(f"{label:20s} как есть {statistics.mean(b):6.1%} (мед. {statistics.median(b):5.1%})  "
              f"улучшено {statistics.mean(e):6.1%} (мед. {statistics.median(e):5.1%})  "
              f"лучше на {sum(1 for x, y in zip(b, e) if y < x)}/{len(ks)}, "
              f"зацикливаний {loops_b} -> {loops_e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
