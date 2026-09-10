"""
Fine-tune Qwen3-VL on Russian handwriting with LoRA.

The base model reads a photographed page at about 31% character error. Its
mistakes are systematic rather than random -- unstressed о read as а, and a
tendency to correct the writer's spelling instead of copying it -- which is
the kind of thing training fixes and prompting does not. The same approach
took our line model from 21.9% to 6.9%.

Trains on HWR200: 200 adult hands, every text photographed scanned, in poor
light and in good light, so the model sees the same words under the lighting
a phone actually produces. Page-level targets come from
scripts/hwr200_pages.py, which shares a document's sentences out across its
pages by line count.

Only the adapter is trained; the base weights stay frozen and quantised, so
this fits a 24 GB card. What comes out is a ~50 MB file the app loads on top
of the same base model it already downloads.

    python scripts/train_qwen.py --data hwr200_pages.jsonl --epochs 1
"""
import argparse
import io
import json
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from PIL import Image
from torch.utils.data import Dataset

MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# The training prompt must be the prompt used at inference, or the model is
# tuned for a question the app never asks.
from ocr_project.page_reader import PROMPT


class Pages(Dataset):
    """Pages straight out of the HWR200 zips -- 44 GB unpacked, so they stay
    packed and are read one entry at a time."""

    def __init__(self, rows, zip_dir, processor, max_pixels):
        self.rows = rows
        self.zip_dir = zip_dir
        self.processor = processor
        self.max_pixels = max_pixels
        self._open = {}

    def __len__(self):
        return len(self.rows)

    def _archive(self, name):
        if name not in self._open:
            self._open[name] = zipfile.ZipFile(os.path.join(self.zip_dir, name))
        return self._open[name]

    def __getitem__(self, i):
        row = self.rows[i]
        data = self._archive(row["zip"]).read(row["image"])
        image = Image.open(io.BytesIO(data)).convert("RGB")
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
        item = {k: v[0] for k, v in batch.items()}

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
    together costs more memory than the gradient accumulation it saves."""
    return {k: v.unsqueeze(0) if v.dim() >= 1 else v for k, v in batch[0].items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hwr200_pages.jsonl")
    ap.add_argument("--zip-dir", default="real_data/HWR200")
    ap.add_argument("--out", default="checkpoints/qwen3vl-hwr200")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=60)
    ap.add_argument("--max-pixels", type=int, default=1280 * 32 * 32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    # Split by writer, not by page: the same hand on both sides of the split
    # would make the score a memory test.
    writers = sorted({r["writer"] for r in rows})
    random.Random(0).shuffle(writers)
    held = set(writers[: max(1, len(writers) // 10)])
    train = [r for r in rows if r["writer"] not in held]
    # The holdout takes the three lighting conditions in equal measure: the
    # whole point is to see whether a photograph in poor light improved, and a
    # holdout that happened to be mostly scans could not show it.
    test = []
    per_condition = max(1, args.holdout // 3)
    for condition in ("Сканы", "ФотоСветлое", "ФотоТемное"):
        test += [r for r in rows if r["writer"] in held
                 and r.get("condition") == condition][:per_condition]
    if args.limit:
        train = train[: args.limit]
    print(f"страниц для обучения {len(train)}, для проверки {len(test)} "
          f"(почерков {len(writers)}, отложено {len(held)})", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)
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

    os.makedirs(args.out, exist_ok=True)
    json.dump(test, open(os.path.join(args.out, "holdout.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=args.accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
            bf16=True, logging_steps=10, save_strategy="steps", save_steps=200,
            save_total_limit=3, report_to=[], remove_unused_columns=False,
            dataloader_num_workers=0, gradient_checkpointing=True,
        ),
        train_dataset=Pages(train, args.zip_dir, processor, args.max_pixels),
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=any(
        d.startswith("checkpoint-") for d in os.listdir(args.out)))
    model.save_pretrained(args.out)
    print(f"адаптер сохранён в {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
