"""
Did the fine-tune help? Same pages, same prompt, base model against adapter.

Run right after training, on writers the adapter never saw. Anything else --
new pages from a hand it was trained on, a different prompt, a different
quantisation -- measures something other than the question asked.

    python scripts/eval_adapter.py --adapter checkpoints/qwen3vl-hwr200
"""
import argparse
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein


def cer(reference: str, hypothesis: str) -> float:
    reference = " ".join(reference.split()).lower()
    hypothesis = " ".join(hypothesis.split()).lower()
    return Levenshtein.distance(reference, hypothesis) / len(reference) if reference else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/qwen3vl-hwr200")
    ap.add_argument("--zip-dir", default="real_data/HWR200")
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--out", default="bench/adapter.tsv")
    args = ap.parse_args()

    import torch
    from ocr_project.page_reader import PageReader

    rows = json.load(open(os.path.join(args.adapter, "holdout.json"), encoding="utf-8"))
    rows = rows[: args.pages]
    print(f"страниц для проверки: {len(rows)} (почерки, которых адаптер не видел)", flush=True)

    archives = {}
    def page(row):
        z = archives.setdefault(row["zip"], zipfile.ZipFile(os.path.join(args.zip_dir, row["zip"])))
        return cv2.imdecode(np.frombuffer(z.read(row["image"]), np.uint8), cv2.IMREAD_COLOR)

    results = {}
    for label, adapter in (("база", None), ("адаптер", args.adapter)):
        reader = PageReader(quantize=True)
        if adapter:
            from peft import PeftModel
            reader.model = PeftModel.from_pretrained(reader.model, adapter)
            reader.model.eval()
        print(f"\n[{label}]", flush=True)
        scores = []
        for i, row in enumerate(rows, 1):
            img = page(row)
            if img is None:
                continue
            text = "\n".join(reader.read_page(img))
            e = cer(row["text"], text)
            scores.append(e)
            results.setdefault(row["image"], {})[label] = e
            results[row["image"]].setdefault("text_" + label, text)
            print(f"  [{i}/{len(rows)}] {e:6.1%}", flush=True)
        reader.model = None
        torch.cuda.empty_cache()
        print(f"  среднее {sum(scores)/len(scores):.1%}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("image\tcer_base\tcer_adapter\n")
        for name, r in results.items():
            f.write(f"{name}\t{r.get('база', float('nan')):.4f}\t{r.get('адаптер', float('nan')):.4f}\n")

    base = [r["база"] for r in results.values() if "база" in r]
    tuned = [r["адаптер"] for r in results.values() if "адаптер" in r]
    if base and tuned:
        import statistics as st
        print(f"\n{'':10s} {'среднее':>9s} {'медиана':>9s}")
        print(f"{'база':10s} {st.mean(base):9.1%} {st.median(base):9.1%}")
        print(f"{'адаптер':10s} {st.mean(tuned):9.1%} {st.median(tuned):9.1%}")
        better = sum(1 for b, t in zip(base, tuned) if t < b)
        print(f"лучше с адаптером на {better} из {len(base)} страниц")
    return 0


if __name__ == "__main__":
    sys.exit(main())
