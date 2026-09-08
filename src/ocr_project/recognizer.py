"""
Page -> lines -> text recognition using TrOCR-ru + our LoRA adapter.

Exposes a single Recognizer class that the GUI drives: it loads the model
once, then turns a page image into a list of RecognizedLine records that
carry both the transcription and a per-line confidence, so the UI can flag
places the model was unsure about.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .rescorer import LanguageRescorer
from .segmentation import segment_lines, split_pages

BASE_MODEL_ID = "raxtemur/trocr-base-ru"
DEFAULT_ADAPTER = "checkpoints/trocr-lora-v5"


@dataclass
class RecognizedLine:
    index: int
    bbox: tuple[int, int, int, int]  # (left, top, right, bottom) in page coords
    text: str
    confidence: float  # 0..1, mean token probability
    image: np.ndarray  # BGR crop, for showing the original beside the text


class Recognizer:
    def __init__(
        self,
        adapter_dir: str | None = DEFAULT_ADAPTER,
        device: str | None = None,
        use_language_model: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(BASE_MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_ID)

        if adapter_dir and os.path.isdir(adapter_dir):
            model = PeftModel.from_pretrained(model, adapter_dir)
            model = model.merge_and_unload()
            self.adapter_loaded = True
        else:
            self.adapter_loaded = False

        self.model = model.to(self.device)
        self.model.eval()

        self.rescorer = None
        if use_language_model and os.path.isdir("models/rugpt3small"):
            self.rescorer = LanguageRescorer(device=self.device)

    def recognize_line(self, line_bgr: np.ndarray, num_beams: int = 5) -> tuple[str, float]:
        image = Image.fromarray(cv2.cvtColor(line_bgr, cv2.COLOR_BGR2RGB))
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        # Ask for every beam's final hypothesis, not just the winner -- the
        # language model below only ever picks among these, it never writes
        # its own text, so it can't introduce a word the OCR model didn't
        # already consider plausible from the image.
        num_return = num_beams if self.rescorer else 1
        with torch.no_grad():
            out = self.model.generate(
                pixel_values,
                # Full lines need far more tokens than the word crops the
                # adapter was trained on; 32 visibly truncated long lines.
                max_new_tokens=96,
                # Beam search keeps several candidate continuations alive
                # instead of committing to the single most-likely token at
                # every step, so it can recover from an early greedy misstep.
                num_beams=num_beams,
                num_return_sequences=num_return,
                output_scores=True,
                return_dict_in_generate=True,
            )

        texts = [t.strip() for t in self.processor.batch_decode(out.sequences, skip_special_tokens=True)]

        if self.rescorer and len(texts) > 1:
            # sort best-first by the OCR model's own score, since
            # best_candidate() trusts index 0 to reflect its verdict on
            # whether the line is blank -- don't rely on generate()'s
            # output order to already guarantee that.
            order = sorted(range(len(texts)), key=lambda i: -out.sequences_scores[i].item())
            candidates = [(texts[i], out.sequences_scores[i].item()) for i in order]
            best_i = order[self.rescorer.best_candidate(candidates)]
        else:
            best_i = 0
        text = texts[best_i]

        # Per-token probability of the sequence actually chosen. With beam
        # search the raw step scores are indexed by beam slot, not by the
        # winning sequence, so compute_transition_scores is used to resolve
        # that (it also works fine for the num_beams=1 / greedy case).
        confidence = 1.0
        beam_indices = getattr(out, "beam_indices", None)
        transition_scores = self.model.compute_transition_scores(
            out.sequences, out.scores, beam_indices, normalize_logits=True
        )
        probs = torch.exp(transition_scores[best_i])
        probs = probs[torch.isfinite(probs)]
        if probs.numel() > 0:
            confidence = float(probs.mean().item())

        return text, confidence

    def recognize_page(self, page_bgr: np.ndarray) -> list[RecognizedLine]:
        # A book/notebook spread photo has two independently-flowing text
        # columns; split_pages() detects that and hands back two images
        # (left/right page) instead of one, so segment_lines() below never
        # has to draw a single row projection across both at once -- a
        # single page photo comes back unsplit and this is a no-op.
        results = []
        i = 0
        for page in split_pages(page_bgr):
            for bbox, crop in segment_lines(page):
                text, confidence = self.recognize_line(crop)
                results.append(
                    RecognizedLine(index=i, bbox=bbox, text=text, confidence=confidence, image=crop)
                )
                i += 1
        return results

    def recognize_file(self, path: str) -> list[RecognizedLine]:
        # np.fromfile + imdecode instead of cv2.imread: imread cannot handle
        # non-ASCII paths on Windows, and this project lives under a Cyrillic
        # directory name.
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {path}")
        return self.recognize_page(image)
