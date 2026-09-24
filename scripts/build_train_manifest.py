"""
The second fine-tune's data: everything it trains on, and the pages it is
judged on, fixed in two files before any GPU time is rented.

Trains on all of HWR200 -- 200 adult hands, every page scanned and
photographed in good and poor light -- and on the school notebooks, whose
children's handwriting is the worst in any open Russian set. Ten writers are
set aside to judge the result, and judged only on texts nobody else wrote.

That last condition comes from looking at the data rather than at its
description. The writers were given topics, and many essays on one topic open
with the same passage: nineteen people start with the same lines about
Esenin. A held-out page carrying one of those passages would be read partly
from memory of another writer's copy, and the score would flatter the model
exactly where the app's users gain nothing.

The 25 benchmark pages are school notebook pages, so the notebooks they come
from -- the same child's hand on other pages -- stay out of training too.

    python scripts/build_train_manifest.py
"""
import argparse
import gzip
import json
import os
import random
import unicodedata
from collections import Counter, defaultdict

ANN_ROOT = "real_data/HWR200/annotations/annotations"
CONDITIONS = ("Сканы", "ФотоСветлое", "ФотоТемное")


def opening(text: str, chars: int = 120) -> str:
    return " ".join(text.split()).lower()[:chars]


def document_openings() -> tuple[dict, dict]:
    """The first words of every text.

    Reuse texts are filed per writer. Original and FPR texts are numbered
    across the whole set -- one file for every writer who copied that text
    out -- so they are looked up by name. Which writers actually wrote a text
    is taken from the photographs, not from the folder its file sits in."""
    per_writer, by_name = {}, {}
    for kind in ("Originals", "Reuse", "FPR"):
        base = os.path.join(ANN_ROOT, kind)
        if not os.path.isdir(base):
            continue
        for folder in os.listdir(base):
            path = os.path.join(base, folder)
            if not os.path.isdir(path):
                continue
            for name in os.listdir(path):
                if not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(path, name), encoding="utf-8-sig") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    continue            # a few dozen annotation files are not valid JSON
                text = data.get("full_text") or " ".join(
                    s.get("text", "") for s in data.get("sentences", []))
                if not text.strip():
                    continue
                document = os.path.splitext(name)[0]
                if kind == "Reuse":
                    per_writer[(folder, document)] = opening(text)
                else:
                    by_name.setdefault(document, opening(text))
    return per_writer, by_name


PARAGRAPH_MARK = "[НОВЫЙ АБЗАЦ]"


def clean_label(text: str, markup: bool = True) -> str:
    """What a hand could have put on the page, and nothing a typist added.

    HWR200's reuse texts carry the annotators' "[НОВЫЙ АБЗАЦ]" -- 8989 times,
    on half the pages. Trained on it, the model learns to write a marker that
    is never on the page, which is inventing text by definition. The rest is
    typing residue: no-break spaces and hyphens, a precomposed ellipsis, й and
    ё split into a letter and a combining mark, stress accents, a BOM.

    Left alone on purpose: quotation marks and angle brackets, where there is
    no telling what the writer drew, and in the notebooks the square brackets
    of phonetic transcription, which the children wrote themselves.
    """
    if markup:
        text = text.replace(PARAGRAPH_MARK, " ")
    text = unicodedata.normalize("NFC", text)
    for old, new in (("́", ""), ("﻿", ""), (" ", " "), ("‑", "-"),
                     ("−", "-"), ("…", "...")):
        text = text.replace(old, new)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def notebook_of(name: str) -> str:
    """School notebook pages are named <notebook>_<page>.jpg."""
    return name.split("_", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hwr", nargs="+", default=["hwr200_pages_all.jsonl"])
    ap.add_argument("--notebooks", default="page_data/school_notebooks_pages.jsonl")
    ap.add_argument("--bench", default="bench/pages.jsonl")
    ap.add_argument("--held-writers", type=int, default=10)
    ap.add_argument("--per-condition", type=int, default=12)
    ap.add_argument("--train-out", default="train_pages.jsonl.gz")
    ap.add_argument("--holdout-out", default="holdout_pages.jsonl")
    ap.add_argument("--rotation-flags", default="rotation_flags.jsonl")
    ap.add_argument("--dedup-cap", type=int, default=None,
                     help="Cap how many documents sharing the same opening "
                          "(see document_openings) go into training. Many "
                          "HWR200 essays answer the same set-topic exam "
                          "prompts and so open with near-identical passages "
                          "-- unbounded, the model can learn to recite one "
                          "of those common openings from memory on a hard "
                          "page instead of reading it. Unset by default: "
                          "changes what round 2 already shipped on.")
    args = ap.parse_args()

    # Pages the app's users never produce: they photograph their own page
    # upright. HWR200 has some shot or scanned sideways instead (no reliable
    # way to tell which way to un-rotate them -- the same page's scan and
    # photo can be sideways in opposite directions), so training on them as-is
    # would just pair the label with a rotated image the model will never see
    # at inference. Dropped rather than fixed.
    rotated = set()
    if os.path.exists(args.rotation_flags):
        with open(args.rotation_flags, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("likely_rotated"):
                    rotated.add((r["zip"], r["image"]))

    rows, seen = [], set()
    marked = 0
    dropped_rotated = 0
    for path in args.hwr:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r["zip"], r["image"])
                if key in seen:
                    continue
                seen.add(key)
                if key in rotated:
                    dropped_rotated += 1
                    continue
                marked += PARAGRAPH_MARK in r["text"]
                r["text"] = clean_label(r["text"])
                if not r["text"]:
                    continue
                r["document"] = r["image"].split("/")[2]
                r["source"] = "hwr200"
                rows.append(r)

    per_writer, by_name = document_openings()

    def opening_of(r):
        return per_writer.get((r["writer"], r["document"])) or by_name.get(r["document"])

    writers_by_opening = defaultdict(set)
    for r in rows:
        op = opening_of(r)
        if op:
            writers_by_opening[op].add(r["writer"])

    def unique_text(r) -> bool:
        op = opening_of(r)
        return op is not None and len(writers_by_opening[op]) == 1

    by_writer = defaultdict(list)
    for r in rows:
        by_writer[r["writer"]].append(r)

    # Writers to set aside: chosen at random, but only among those with enough
    # pages of their own texts in every lighting to be judged on.
    writers = sorted(by_writer)
    random.Random(0).shuffle(writers)
    held = []
    for w in writers:
        if all(sum(1 for r in by_writer[w] if r["condition"] == c and unique_text(r)) >= 2
               for c in CONDITIONS):
            held.append(w)
        if len(held) == args.held_writers:
            break
    held_set = set(held)

    # Pages to judge on: round-robin over the held-out writers in each
    # lighting, single-page documents first -- their text is exact, where a
    # longer document's text is shared out across its pages by estimate.
    holdout = []
    for condition in CONDITIONS:
        queues = []
        for w in held:
            pages = [r for r in by_writer[w] if r["condition"] == condition and unique_text(r)]
            pages.sort(key=lambda r: (not r.get("exact"), r["image"]))
            queues.append(pages)
        picked = []
        while len(picked) < args.per_condition and any(queues):
            for q in queues:
                if q and len(picked) < args.per_condition:
                    picked.append(q.pop(0))
        holdout += picked

    train = [r for r in rows if r["writer"] not in held_set]
    set_aside = len(rows) - len(train) - len(holdout)

    dropped_dup = 0
    if args.dedup_cap is not None:
        # Group by document first (opening_of is a per-document property, but
        # capping per-page would just thin out one document's own pages
        # rather than removing whole duplicate copies), then by shared
        # opening across documents.
        docs = defaultdict(list)
        for r in train:
            docs[(r["writer"], r["document"])].append(r)
        by_opening = defaultdict(list)
        for key, doc_rows in docs.items():
            op = opening_of(doc_rows[0])
            by_opening[op].append(key)  # op is None for a handful of unmatched documents; left uncapped
        kept_docs = set()
        for op, keys in by_opening.items():
            if op is None or len(keys) <= args.dedup_cap:
                kept_docs.update(keys)
            else:
                kept_docs.update(sorted(keys)[:args.dedup_cap])
        new_train = [r for r in train if (r["writer"], r["document"]) in kept_docs]
        dropped_dup = len(train) - len(new_train)
        train = new_train

    with open(args.bench, encoding="utf-8") as f:
        bench_names = {os.path.basename(json.loads(l)["image"].replace("\\", "/")) for l in f}
    bench_notebooks = {notebook_of(n) for n in bench_names}
    notebooks_total = notebooks_kept = 0
    with open(args.notebooks, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            name = os.path.basename(r["image"].replace("\\", "/"))
            notebooks_total += 1
            if notebook_of(name) in bench_notebooks:
                continue
            text = clean_label(r["text"], markup=False)
            if text:
                train.append({"source": "notebooks", "file": name, "text": text})
                notebooks_kept += 1

    random.Random(0).shuffle(train)
    with gzip.open(args.train_out, "wt", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.holdout_out, "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sources = Counter(r["source"] for r in train)
    print(f"HWR200: страниц {len(rows)}, почерков {len(by_writer)}, "
          f"пометка абзаца убрана на {marked} страницах, "
          f"выброшено как повёрнутые {dropped_rotated}")
    if args.dedup_cap is not None:
        print(f"дедупликация (--dedup-cap {args.dedup_cap}): выброшено страниц "
              f"дублирующихся документов {dropped_dup}")
    print(f"  в обучение {sources['hwr200']}, на проверку {len(holdout)} "
          f"(почерков {len(held)}), не взято {set_aside} -- страницы отложенных почерков "
          f"с текстами, которые писал кто-то ещё")
    print(f"  проверка по условиям: {dict(Counter(r['condition'] for r in holdout))}, "
          f"точных {sum(1 for r in holdout if r.get('exact'))}")
    print(f"тетради: {notebooks_total}, в обучение {notebooks_kept}, "
          f"убрано вместе с проверочными тетрадями {notebooks_total - notebooks_kept}")
    print(f"итого в обучение: {len(train)} -> {args.train_out}")
    print(f"отложенные почерки: {', '.join(held)} -> {args.holdout_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
