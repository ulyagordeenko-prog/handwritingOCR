"""
Turn HWR200 into page-level training pairs.

The dataset gives one text per document and one image per page: a five-page
essay comes with 3359 characters and no indication of where page one ends.
Training on that directly would teach the model to read text that is not on
the page in front of it.

The split is done by counting text lines on each page -- the segmentation the
app already uses -- and handing out the document's sentences in proportion to
those counts, cutting only at sentence boundaries. A writer's lines hold
roughly the same number of characters throughout one document, so line counts
carry the proportions; sentence boundaries keep the error at a page break to
at most part of one sentence rather than a torn word.

Exact where it can be: a single-page document takes the whole text with no
estimation at all, and those pages are marked so training can weight them.

    python scripts/hwr200_pages.py --zip real_data/HWR200/hwr200_0_19.zip
"""
import argparse
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

ANN_ROOT = "real_data/HWR200/annotations/annotations"
CONDITIONS = ("Сканы", "ФотоСветлое", "ФотоТемное")


def count_lines(data: bytes, probe_width: int = 900) -> int:
    """How many text lines are on this page. Deliberately cheap: the image is
    read at a reduced size, because only the count matters, not the crops."""
    from ocr_project.segmentation import segment_lines

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return 0
    h, w = image.shape[:2]
    if w > probe_width:
        image = cv2.resize(image, (probe_width, int(h * probe_width / w)),
                           interpolation=cv2.INTER_AREA)
    try:
        return len(segment_lines(image, target_width=probe_width))
    except Exception:
        return 0


def load_annotation(writer: str, document: str) -> list[str] | None:
    """The sentences of one document, in order."""
    for kind in ("Originals", "Reuse", "FPR"):
        path = os.path.join(ANN_ROOT, kind, writer, f"{document}.json")
        if os.path.isfile(path):
            data = json.load(open(path, encoding="utf-8"))
            sentences = [s["text"].strip() for s in data.get("sentences", []) if s.get("text")]
            if sentences:
                return sentences
            text = (data.get("full_text") or "").strip()
            return re.split(r"(?<=[.!?])\s+", text) if text else None
    return None


def share_out(sentences: list[str], counts: list[int]) -> list[str]:
    """Give each page a run of whole sentences, sized by its share of lines."""
    total_lines = sum(counts)
    if total_lines == 0:
        return [""] * len(counts)
    lengths = [len(s) + 1 for s in sentences]
    total_chars = sum(lengths)

    pages, taken, used = [], 0, 0
    for i, count in enumerate(counts):
        if i == len(counts) - 1:
            pages.append(" ".join(sentences[taken:]))
            break
        used += count
        target = total_chars * used / total_lines
        running = sum(lengths[:taken])
        end = taken
        # take whole sentences until this page holds its share
        while end < len(sentences) and running + lengths[end] / 2 < target:
            running += lengths[end]
            end += 1
        pages.append(" ".join(sentences[taken:end]))
        taken = end
    return pages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="hwr200_pages.jsonl")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--limit", type=int, default=0, help="0 = без ограничения")
    args = ap.parse_args()
    wanted = set(args.conditions.split(","))

    archive = zipfile.ZipFile(args.zip)
    groups = defaultdict(list)
    for name in archive.namelist():
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = name.split("/")
        if len(parts) < 5 or parts[3] not in wanted:
            continue
        groups[(parts[1], parts[2], parts[3])].append(name)

    print(f"документов в архиве: {len(groups)}", flush=True)
    written = exact = skipped = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i, (key, names) in enumerate(sorted(groups.items()), 1):
            writer, document, condition = key
            sentences = load_annotation(writer, document)
            if not sentences:
                skipped += 1
                continue
            names.sort(key=lambda n: int(re.sub(r"\D", "", n.split("/")[-1]) or 0))
            counts = [count_lines(archive.read(n)) for n in names]
            if sum(counts) == 0:
                skipped += 1
                continue
            texts = share_out(sentences, counts)
            for name, text, count in zip(names, texts, counts):
                if not text.strip():
                    continue
                out.write(json.dumps({
                    "zip": os.path.basename(args.zip), "image": name,
                    "text": text, "lines": count, "pages_in_doc": len(names),
                    "exact": len(names) == 1, "writer": writer,
                    "condition": condition,
                }, ensure_ascii=False) + "\n")
                written += 1
                exact += len(names) == 1
            if i % 25 == 0:
                print(f"  {i}/{len(groups)} документов, страниц записано {written}", flush=True)
            if args.limit and written >= args.limit:
                break

    print(f"\nзаписано страниц: {written} (точных {exact}), пропущено документов: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
