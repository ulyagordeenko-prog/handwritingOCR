"""
Reading with PaddleOCR-VL: 0.9 billion parameters against Qwen3-VL's eight.

Worth measuring because the ranking on document benchmarks is not the ranking
by size -- compact models built specifically for document parsing sit above
general-purpose VLMs there. If that holds for Russian handwriting, the app
stops needing an 8 GB card at all: 2 GB of weights run anywhere, and the model
reads a line in a fraction of the time Qwen needs for a page.

It is an element-level recogniser, not a page reader: the intended pipeline
feeds it one detected block at a time. We already cut pages into lines, so
that is the mode to test -- but a whole page is tried too, since the model was
trained on document layout and might handle one directly.
"""
from __future__ import annotations

import re

import torch
from PIL import Image

MODEL_NAME = "PaddlePaddle/PaddleOCR-VL"

# The prompt is not free text. The processor's own chat template wraps it as
# "<|begin_of_sentence|>User: <image>OCR:\nAssistant: ", and a hand-written
# prompt missing that framing produces an empty answer with no error at all --
# which is exactly what a first attempt did.
TASK = "OCR:"

# Table structure markup the model emits on tabular blocks; we want the text.
_MARKUP = re.compile(r"<(?:[efluxn]cel|nl)>")


class PaddleReader:
    def __init__(self, device: str | None = None, dtype=None):
        from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            MODEL_NAME, dtype=dtype
        ).to(self.device).eval()

    def _read(self, image_bgr, max_new_tokens: int = 256) -> str:
        import cv2

        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": TASK}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return _MARKUP.sub(" ", text).strip()

    def read_lines(self, page_bgr, progress=None) -> list[str]:
        """The mode this model is built for: one text block per call."""
        from .segmentation import crop_to_page, deskew, segment_lines, split_pages

        page_bgr, _ = crop_to_page(page_bgr)
        page_bgr, _ = deskew(page_bgr)
        crops = [c for page in split_pages(page_bgr) for _, c in segment_lines(page)]

        lines = []
        for i, crop in enumerate(crops):
            if progress:
                progress(i, len(crops))
            text = self._read(crop, max_new_tokens=128)
            if text:
                lines.append(text)
        if progress:
            progress(len(crops), len(crops))
        return lines

    def read_page(self, page_bgr, progress=None) -> list[str]:
        """A whole page in one call -- outside the intended use, measured anyway."""
        from .segmentation import crop_to_page, deskew, split_pages

        page_bgr, _ = crop_to_page(page_bgr)
        page_bgr, _ = deskew(page_bgr)
        pages = split_pages(page_bgr)

        lines = []
        for i, page in enumerate(pages):
            if progress:
                progress(i, len(pages))
            lines.extend(l for l in self._read(page, max_new_tokens=1536).splitlines() if l.strip())
        if progress:
            progress(len(pages), len(pages))
        return lines

    def unload(self):
        """Free the card before the next model is loaded onto it."""
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
