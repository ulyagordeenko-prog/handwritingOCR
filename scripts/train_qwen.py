"""
Fine-tune Qwen3-VL on Russian handwriting with LoRA.

The base model reads a photographed page at about 31% character error. Its
mistakes are systematic rather than random -- unstressed о read as а, and a
tendency to correct the writer's spelling instead of copying it -- which is
the kind of thing training fixes and prompting does not. The same approach
took our line model from 21.9% to 6.9%.

The second run trains on all of HWR200 -- 200 adult hands, every text scanned
and photographed in poor and in good light -- plus the school notebooks, and
nothing the evaluation is later run on (scripts/build_train_manifest.py
decides that, once, before the rental). The first run saw 40 hands; on
held-out pages it helped scans and did nothing for bright photos.

Only the adapter is trained; the base weights stay frozen and quantised.
What comes out is a ~60 MB file the app loads on top of the same base model
it already downloads.

    python scripts/build_train_manifest.py        # locally, once
    python scripts/train_qwen.py --check-data 20  # data and labels, no GPU
    python scripts/train_qwen.py --smoke 12       # will the heaviest pages fit?
    python scripts/train_qwen.py                  # the run; resumes if stopped
"""
import argparse
import gzip
import io
import json
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset

MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Fields shaped (batch, sequence). Everything else the processor returns --
# pixel_values, image_grid_thw -- is laid out per image patch and must pass
# through untouched.
TOKEN_FIELDS = ("input_ids", "attention_mask", "mm_token_type_ids")

# The training prompt must be the prompt used at inference, or the model is
# tuned for a question the app never asks.
from ocr_project.page_reader import PROMPT


def read_rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def roughen(image, rng):
    """A worse photograph of the same page.

    Mild on purpose: HWR200 already brings every page in three lightings.
    This adds what a hurried phone shot puts on top -- a little blur, a shift
    in exposure, a degree or two of tilt, heavy JPEG. The text is unchanged;
    only how hard it is to see."""
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    if rng.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.15))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.2))
    if rng.random() < 0.3:
        image = image.rotate(rng.uniform(-2, 2), resample=Image.BICUBIC, expand=True,
                             fillcolor=image.getpixel((0, 0)))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=rng.randint(35, 70))
        image = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return image


class Pages(Dataset):
    """HWR200 pages straight out of the zips -- 44 GB unpacked, so they stay
    packed and are read one entry at a time -- and notebook pages from disk."""

    def __init__(self, rows, zip_dir, notebook_dir, processor, max_pixels, augment=0.0, seed=0):
        self.rows = rows
        self.zip_dir = zip_dir
        self.notebook_dir = notebook_dir
        self.processor = processor
        self.max_pixels = max_pixels
        self.augment = augment
        self.rng = random.Random(seed)
        self._open = {}

    def __len__(self):
        return len(self.rows)

    def _archive(self, name):
        if name not in self._open:
            self._open[name] = zipfile.ZipFile(os.path.join(self.zip_dir, name))
        return self._open[name]

    def load(self, row):
        if row.get("source") == "notebooks":
            image = Image.open(os.path.join(self.notebook_dir, row["file"]))
        else:
            image = Image.open(io.BytesIO(self._archive(row["zip"]).read(row["image"])))
        # Phone photos store their rotation as a tag. The app reads through
        # OpenCV, which applies it; without this the model would train on
        # sideways pages the app never shows it.
        return ImageOps.exif_transpose(image).convert("RGB")

    def __getitem__(self, i):
        row = self.rows[i]
        image = self.load(row)
        if self.augment and self.rng.random() < self.augment:
            image = roughen(image, self.rng)
        w, h = image.size
        if w * h > self.max_pixels:                 # keep the token count bounded
            k = (self.max_pixels / (w * h)) ** 0.5
            image = image.resize((int(w * k), int(h * k)), Image.LANCZOS)

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": row["text"]}]},
        ]
        batch = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
        )
        # Only the token fields carry a batch dimension. pixel_values is
        # (patches, features) and image_grid_thw is (images, 3); taking [0] of
        # those -- which the first version of this did for every field -- keeps
        # one image patch out of ~5000 and throws the rest of the page away.
        item = {k: (v[0] if k in TOKEN_FIELDS else v) for k, v in batch.items()}

        # Loss on the answer only: the prompt and the image are the question,
        # and training the model to predict its own question teaches nothing.
        prompt_only = self.processor.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        cut = prompt_only["input_ids"].shape[-1]
        labels = item["input_ids"].clone()
        labels[:cut] = -100
        item["labels"] = labels
        return item


def collate(batch):
    """One sample per step: pages differ in size, and padding image grids
    together costs more memory than the gradient accumulation it saves.
    Token fields get their batch dimension back; image fields never lost it."""
    return {k: (v.unsqueeze(0) if k in TOKEN_FIELDS + ("labels",) else v)
            for k, v in batch[0].items()}


def smoke(model, dataset, steps) -> int:
    """Put the heaviest pages through real training steps and report the peak
    memory, before hours are committed to a card that may not hold them."""
    model.train()
    torch.cuda.reset_peak_memory_stats()
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    try:
        for i in range(min(steps, len(dataset))):
            batch = {k: v.to("cuda") for k, v in collate([dataset[i]]).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  шаг {i + 1}: loss {loss.item():.3f}, пик {peak:.1f} из {total:.1f} ГБ", flush=True)
    except torch.cuda.OutOfMemoryError:
        print("НЕ_ХВАТИЛО_ПАМЯТИ", flush=True)
        return 3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    # the optimiser's state and allocator fragmentation come on top in the run
    if peak + 1.5 > total:
        print(f"ВПРИТЫК {peak:.1f} из {total:.1f} ГБ", flush=True)
        return 4
    print(f"ПАМЯТИ_ХВАТАЕТ {peak:.1f} из {total:.1f} ГБ", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_pages.jsonl.gz")
    ap.add_argument("--holdout", default="holdout_pages.jsonl")
    ap.add_argument("--zip-dir", default="real_data/HWR200")
    ap.add_argument("--notebook-dir", default="real_data/school_notebooks_RU/images")
    ap.add_argument("--out", default="checkpoints/qwen3vl-hwr200-full")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1280 * 32 * 32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--augment", type=float, default=0.3)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--smoke", type=int, default=0, help="N тяжелейших страниц, затем выход")
    ap.add_argument("--check-data", type=int, default=0, help="N страниц без видеокарты")
    args = ap.parse_args()

    from transformers import AutoProcessor

    train = read_rows(args.data)
    # The manifest already keeps the held-out writers out; this is the check
    # that it did, because a leak here would make every later number a lie.
    if os.path.exists(args.holdout):
        held = {json.loads(l)["writer"] for l in open(args.holdout, encoding="utf-8")}
        leaked = sum(1 for r in train if r.get("writer") in held)
        if leaked:
            raise SystemExit(f"в обучение попали отложенные почерки: {leaked} страниц")
    if args.limit:
        train = train[: args.limit]
    by_source = {}
    for r in train:
        by_source[r.get("source", "?")] = by_source.get(r.get("source", "?"), 0) + 1
    print(f"страниц для обучения {len(train)}: {by_source}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)

    if args.check_data:
        sample = random.Random(1).sample(train, min(args.check_data, len(train)))
        ds = Pages(sample, args.zip_dir, args.notebook_dir, processor, args.max_pixels, augment=1.0)
        for i, row in enumerate(sample):
            item = ds[i]
            answer = int((item["labels"] != -100).sum())
            decoded = processor.tokenizer.decode(item["input_ids"][item["labels"] != -100][:40])
            print(f"  {row.get('source'):9s} токенов {item['input_ids'].shape[-1]:5d}, "
                  f"из них ответ {answer:4d}, картинка {tuple(item['image_grid_thw'][0].tolist())} "
                  f"| {decoded[:60]!r}", flush=True)
        print("ДАННЫЕ_В_ПОРЯДКЕ", flush=True)
        return 0

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()

    if args.smoke:
        heaviest = sorted(train, key=lambda r: -len(r["text"]))[: args.smoke]
        return smoke(model, Pages(heaviest, args.zip_dir, args.notebook_dir, processor,
                                  args.max_pixels), args.smoke)

    os.makedirs(args.out, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=args.accum,
            # transformers 5 dropped warmup_ratio; the same 3% expressed in steps
            learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_steps=max(1, int(0.03 * len(train) * args.epochs / args.accum)),
            # Saved about every 25 minutes: a rental that ends or a link that
            # drops costs that much, not the hours before it.
            bf16=True, logging_steps=10, save_strategy="steps", save_steps=args.save_steps,
            save_total_limit=3, report_to=[], remove_unused_columns=False,
            dataloader_num_workers=0, gradient_checkpointing=True,
        ),
        train_dataset=Pages(train, args.zip_dir, args.notebook_dir, processor, args.max_pixels,
                            augment=args.augment),
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=any(
        d.startswith("checkpoint-") for d in os.listdir(args.out)))
    model.save_pretrained(args.out)
    print(f"адаптер сохранён в {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
