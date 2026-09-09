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

import math

import torch
from PIL import Image

from .segmentation import crop_to_page, deskew, split_pages, split_strips

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# The task is transcription, not comprehension: the model must copy what the
# strokes say, the way a person reading someone else's handwriting does, and
# never substitute a word it finds more plausible. Hence the explicit ban on
# replacing an unclear word with a similar one -- guessing is exactly the
# failure mode that makes an otherwise low error rate untrustworthy, because a
# confident wrong word looks the same as a right one.
#
# Note what is NOT asked for: marking unreadable spots with a placeholder. That
# was tried, and on a faded page the model emitted hundreds of them in a row;
# with nothing to repeat, it stops looping.
PROMPT = (
    "Перепиши рукописный текст с этого изображения, строка за строкой. "
    "Переписывай ровно то, что видишь, буква за буквой, не исправляя и не додумывая. "
    "Не заменяй непонятное слово похожим или более подходящим по смыслу — "
    "пиши то, что видно по буквам, даже если получается странно. "
    "Бледные следы текста, просвечивающие с обратной стороны листа, пропускай. "
    "Ничего не повторяй дважды. "
    # The page is read in pieces, so a preamble is not merely noise: it is
    # repeated once per piece and lands in the middle of the transcript.
    "Не пиши ничего, кроме самого текста: ни вступлений, ни пояснений."
)

# rough floor: 4-bit weights are ~5-6 GB, plus room for the image tokens
MIN_VRAM_BYTES = 7 * 1024**3

# Qwen3-VL turns a picture into visual tokens of 32x32 pixels and caps how many
# it will make. Anything above the cap is downscaled before the model ever looks
# at it -- our 1935x1960 test photo loses two thirds of its area that way, and
# no amount of sharpening survives that. Reading the page in horizontal strips
# keeps each piece under the cap, so the model sees the original pixels.
MAX_VISUAL_TOKENS = 1280
PIXELS_PER_TOKEN = 32 * 32
MAX_PIXELS = MAX_VISUAL_TOKENS * PIXELS_PER_TOKEN
# A tall photo needs more pieces than a square one. Three was too few: a
# 6000x2000 shot still came out downscaled 3.3x, which defeats the purpose.
MAX_STRIPS = 6


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

    def read_page(self, page_bgr, max_new_tokens: int = 1024, strips: bool = True,
                  progress=None) -> list[str]:
        """Read a whole page. With `strips`, the page is cropped to the sheet,
        straightened, and cut into as many horizontal pieces as it takes to stay
        under the model's pixel budget -- then each piece is read at full
        resolution and the results are concatenated in reading order."""
        if strips:
            page_bgr, _ = crop_to_page(page_bgr)
            page_bgr, _ = deskew(page_bgr)
            pieces = []
            # An open notebook photographed as a spread must be separated
            # first. Horizontal strips cut across both pages at once, so each
            # piece would hold two unrelated columns and the model would read
            # them as one interleaved text.
            for page in split_pages(page_bgr):
                h, w = page.shape[:2]
                count = min(MAX_STRIPS, max(1, math.ceil((h * w) / MAX_PIXELS)))
                pieces.extend(split_strips(page, count))
        else:
            pieces = [page_bgr]

        lines: list[str] = []
        for i, piece in enumerate(pieces):
            if progress:
                progress(i, len(pieces))
            lines.extend(self._read_one(piece, max_new_tokens))
        if progress:
            progress(len(pieces), len(pieces))
        return lines

    def _read_one(self, page_bgr, max_new_tokens: int = 1024) -> list[str]:
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

    def read_file(self, path: str, strips: bool = True, progress=None) -> list[str]:
        import cv2
        import numpy as np

        # imread can't open non-ASCII paths on Windows
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"не удалось прочитать изображение: {path}")
        return self.read_page(image, strips=strips, progress=progress)
