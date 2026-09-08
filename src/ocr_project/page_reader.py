"""
Whole-page reading with Qwen3-VL.

Unlike the line-by-line recognizer, this looks at the entire photo at once
and uses its own understanding of language to work out what is written --
it needs no line segmentation, keeps reading order on its own, and ignores
the faint bleed-through that the line pipeline picks up as phantom rows.

Measured on a rented RTX 3090 with no fine-tuning at all: 4.0% character
error on a notebook page, against 6.9% for our fine-tuned line model.

The weights are ~17 GB in full precision, so they are loaded quantized to
4 bits (about 5-6 GB) to fit an 8 GB laptop card.
"""
from __future__ import annotations

import torch
from PIL import Image

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Asking it to mark unreadable spots with a placeholder made it emit
# hundreds of them in a row on a faded page; telling it to skip such marks
# instead gives it nothing to loop on.
PROMPT = (
    "Перепиши разборчивый рукописный текст с этого изображения, строка за строкой. "
    "Переписывай ровно то, что видишь, не исправляя и не додумывая. "
    "Бледные следы текста, просвечивающие с обратной стороны листа, пропускай. "
    "Ничего не повторяй дважды."
)

# rough floor: 4-bit weights are ~5-6 GB, plus room for the image tokens
MIN_VRAM_BYTES = 7 * 1024**3


def enough_vram() -> bool:
    """Can this machine realistically run Qwen at all?

    is_available() only says a driver and a card are present; on a card
    newer than the installed torch build (an RTX 50-series against a cu121
    wheel) every kernel launch still fails, so reuse the recognizer's
    probe, which actually runs an operation before saying yes."""
    from .recognizer import _usable_device

    device, _ = _usable_device()
    if device != "cuda":
        return False
    try:
        return torch.cuda.get_device_properties(0).total_memory >= MIN_VRAM_BYTES
    except Exception:
        return False


class PageReader:
    def __init__(self, quantize: bool = True):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kwargs = {"dtype": torch.bfloat16, "device_map": "cuda"}
        if quantize:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, **kwargs)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def read_page(self, page_bgr, max_new_tokens: int = 1024) -> list[str]:
        import cv2

        image = Image.fromarray(cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}],
        }]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        return [line for line in text.splitlines() if line.strip()]

    def read_file(self, path: str) -> list[str]:
        import cv2
        import numpy as np

        # imread can't open non-ASCII paths on Windows
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"не удалось прочитать изображение: {path}")
        return self.read_page(image)
