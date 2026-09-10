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

# Reading the page one line at a time was tried and measured worse than
# reading it whole: 40.5% character error against 15.9% on the same six pages,
# and nearly twice as slow. The idea came from an earlier 4.0% measured on
# individual lines, but those were the dataset's own clean line images; our
# segmentation is not that, and reading a line in isolation strips the context
# that lets the model settle abbreviations and case. Not kept.

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


def trim_runaway(lines: list[str], max_repeats: int = 3) -> list[str]:
    """Cut a degenerate repetition loop out of the model's output.

    Greedy decoding sometimes falls into a rut and emits one fragment until it
    runs out of budget -- measured on a real page: "снега, снега, снега..."
    several hundred times, turning a 1349-character page into 2364 characters
    of output and a character error above 100%. It happens on strips more than
    on whole pages, but it happens on both, and those runaways dominate the
    average error: excluding them drops whole-page error from 38.9% to 24.9%.

    Blanket repetition bans are not an option. The reference transcript of that
    same page legitimately contains "много яблок - много помидоров" twice in a
    row, because that is what the child wrote. So this cuts only what no one
    writes by hand: the same short fragment more than `max_repeats` times in
    immediate succession. Everything after the cut on that line is dropped,
    since a model in a loop has stopped reading the page.
    """
    cleaned = []
    for line in lines:
        cleaned.append(_trim_line(line, max_repeats))

    # the same collapse can span whole lines rather than sit inside one
    out: list[str] = []
    for line in cleaned:
        run = 1
        while run < len(out) + 1 and out[len(out) - run:] == [line] * run:
            run += 1
        if run - 1 >= max_repeats and line.strip():
            continue
        out.append(line)
    return out


def _trim_line(line: str, max_repeats: int) -> str:
    parts = [p for p in line.split(",")]
    if len(parts) <= max_repeats:
        return line
    keep, run = [], 1
    for i, part in enumerate(parts):
        if i and part.strip() == parts[i - 1].strip() and part.strip():
            run += 1
        else:
            run = 1
        if run > max_repeats:
            break
        keep.append(part)
    # A line with nothing to cut is returned untouched. Reassembling it would
    # silently normalise it -- an earlier version stripped the trailing comma
    # from every list line, which shows up as a character error on text that
    # was transcribed perfectly.
    if len(keep) == len(parts):
        return line
    return ",".join(keep).rstrip(", ")


def _is_filler(text: str) -> bool:
    """One character repeated is not handwriting.

    Observed verbatim in line-mode output on blank ruled bands: a line of
    dashes and a line of ninety zeros. Deliberately narrow -- a passport
    number is digits and must survive, so only a single repeated character
    with no variety at all is dropped."""
    stripped = "".join(text.split())
    return len(stripped) >= 4 and len(set(stripped)) == 1


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

    def read_page(self, page_bgr, max_new_tokens: int = 1024, strips: bool = False,
                  progress=None) -> list[str]:
        """Read a photographed page.

        Cropping to the sheet, straightening, and separating a two-page spread
        always happen: an open notebook photographed as a spread otherwise
        reaches the model as two interleaved columns, and separating them was
        worth 54% -> 12% character error on one measured page.

        Cutting each page into horizontal strips is off by default. It was
        built to stop the model downscaling a 3.8-megapixel photo into its
        1.3-megapixel token budget, and it does that -- but measured over six
        pages it changed nothing that matters: 16.1% mean character error
        against 15.9% for whole pages, with the medians favouring strips by as
        little as the means favour whole. A tie is not a reason to add three
        generations per page, so the plain path is the default and this stays
        as a switch.
        """
        page_bgr, _ = crop_to_page(page_bgr)
        page_bgr, _ = deskew(page_bgr)

        pieces = []
        for page in split_pages(page_bgr):
            if not strips:
                pieces.append(page)
                continue
            h, w = page.shape[:2]
            count = min(MAX_STRIPS, max(1, math.ceil((h * w) / MAX_PIXELS)))
            pieces.extend(split_strips(page, count))

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
        return trim_runaway([line for line in text.splitlines() if line.strip()])

    def read_file(self, path: str, strips: bool = False, progress=None) -> list[str]:
        import cv2
        import numpy as np

        # imread can't open non-ASCII paths on Windows
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"не удалось прочитать изображение: {path}")
        return self.read_page(image, strips=strips, progress=progress)
