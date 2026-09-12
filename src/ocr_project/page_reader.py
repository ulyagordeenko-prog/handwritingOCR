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

import os
import re

import torch
from PIL import Image

from .segmentation import crop_to_page, deskew, split_pages

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# LoRA adapter trained on HWR200: 200 adult hands, each text scanned and
# photographed in good and poor light. Loaded on top of the base model when
# present; the app works without it, and whether it ships is decided by the
# held-out measurement, not by its existence.
DEFAULT_ADAPTER = "checkpoints/qwen3vl-hwr200"

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

# Below that floor the model can still be read from ordinary memory, a layer
# at a time -- but 16-bit weights are ~16 GB, so the memory has to be there.
MIN_RAM_GB_SPLIT = 20


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
    of output and a character error above 100%. Those runaways dominate the
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
        cleaned.append(_trim_line(_collapse_loops(line), max_repeats))

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


# Any run of eight or more characters repeated three times or more, whatever
# separates the copies. The first version of this split on commas alone, which
# caught "снега, снега, снега..." and missed the worse case: a whole sentence
# repeated through periods -- "На земле листья окрасились в яркие цвета." some
# twenty times, 2183 characters on one line against a 810-character page.
# Three copies, not two: a page's reference transcript legitimately contains
# "много яблок - много помидоров" twice running, because the child wrote it
# twice.
_LOOP = re.compile(r"(.{8,}?)\1{2,}", re.DOTALL)


def _collapse_loops(line: str) -> str:
    previous = None
    while previous != line:                 # a loop can nest inside a loop
        previous = line
        line = _LOOP.sub(lambda m: m.group(1), line)
    return line


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


def _ram_gb() -> float:
    """How much memory this machine has, in gigabytes."""
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / 1024**3
    except Exception:
        return 0.0


def can_split() -> bool:
    """Can a card too small for the model still read a page, with the rest of
    the model kept in ordinary memory?

    A card is still needed -- the weights are copied into it layer by layer --
    and enough memory to hold the ones waiting their turn.
    """
    from .recognizer import _usable_device

    device, _ = _usable_device()
    return device == "cuda" and _ram_gb() >= MIN_RAM_GB_SPLIT


def _split_across(gpu_reserve_gb: float = 1.0) -> dict:
    """How much of the model each device may hold."""
    free = 0.0
    try:
        free = torch.cuda.mem_get_info()[0] / 1024**3
    except Exception:
        pass
    gpu = max(1, int(free - gpu_reserve_gb))
    cpu = max(8, int(_ram_gb()) - 8)
    return {0: f"{gpu}GiB", "cpu": f"{cpu}GiB"}


class PageReader:
    def __init__(self, quantize: bool = True, adapter_dir: str | None = DEFAULT_ADAPTER,
                 offload: bool = False):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if offload:
            # The model lives in ordinary memory and is copied into the card a
            # layer at a time. Quantizing it as well does not work: with 4-bit
            # weights the layers left behind arrive as empty placeholders and
            # generation dies on the first of them ("cannot copy out of meta
            # tensor"). Plain 16-bit both loads and runs -- and measured twice
            # as fast as 32-bit on the small card this was tried on.
            kwargs = {"dtype": torch.float16, "device_map": "auto",
                      "max_memory": _split_across()}
        else:
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

        self.adapter_loaded = False
        if adapter_dir and os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            self.model.eval()
            self.adapter_loaded = True

    def read_page(self, page_bgr, max_new_tokens: int = 1024, progress=None) -> list[str]:
        """Read a photographed page.

        The photo is cropped to the sheet, straightened, and separated into
        two pages if it is a spread -- an open notebook otherwise reaches the
        model as two interleaved columns, and separating them was worth 54% ->
        12% character error on one measured page.

        Cutting each page into horizontal strips was built and then removed.
        It does what it claimed -- a 3.8-megapixel photo no longer gets
        downscaled into the model's 1.3-megapixel token budget -- but over six
        measured pages that bought nothing: 16.1% character error against
        15.9% for plain whole pages. It cost three generations per page and
        four bugs, so it is gone rather than left as an unused switch.
        """
        page_bgr, _ = crop_to_page(page_bgr)
        page_bgr, _ = deskew(page_bgr)
        pieces = split_pages(page_bgr)

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
            # not "cuda": with part of the model in RAM the first layer decides
            # where the input goes, and accelerate moves it on from there
        ).to(getattr(self.model, "device", "cuda"))

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        return trim_runaway([line for line in text.splitlines() if line.strip()])

    def read_region(self, region_bgr, progress=None) -> list[str]:
        """Read exactly the piece someone drew a box around.

        None of read_page's preparation happens here. Cropping and
        straightening look for a sheet inside a photograph, and a selection is
        already the part that matters; spread splitting would be worse than
        useless, cutting a wide stamp down the middle because its shape looks
        like an open book.
        """
        if progress:
            progress(0, 1)
        lines = self._read_one(region_bgr)
        if progress:
            progress(1, 1)
        return lines

    def read_file(self, path: str, progress=None) -> list[str]:
        import cv2
        import numpy as np

        # imread can't open non-ASCII paths on Windows
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"не удалось прочитать изображение: {path}")
        return self.read_page(image, progress=progress)
