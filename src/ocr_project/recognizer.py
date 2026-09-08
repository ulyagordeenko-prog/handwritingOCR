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


def _usable_device() -> tuple[str, str | None]:
    """Pick cuda only if this build can actually run on this card.

    torch.cuda.is_available() answers "is there a driver and a GPU", not
    "was this wheel compiled for this GPU". A card newer or older than the
    wheel's compiled architectures passes that check and then dies at the
    first real kernel launch with "no kernel image is available for
    execution on the device". Launching a tiny op here turns that crash
    mid-page into a quiet fall back to CPU.

    Returns (device, warning) -- warning is None when all is well.
    """
    if not torch.cuda.is_available():
        return "cpu", None
    try:
        torch.zeros(8, device="cuda").sum().item()
        return "cuda", None
    except Exception as exc:
        name = "неизвестная"
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return "cpu", (
            f"Видеокарта {name} не поддерживается установленной версией PyTorch "
            f"({torch.__version__}), поэтому распознавание идёт на процессоре — "
            f"это работает, но медленнее.\n\nПодробность: {exc}"
        )


class Recognizer:
    def __init__(
        self,
        adapter_dir: str | None = DEFAULT_ADAPTER,
        device: str | None = None,
        use_language_model: bool = True,
    ):
        if device:
            self.device, self.device_warning = device, None
        else:
            self.device, self.device_warning = _usable_device()
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
        return self.recognize_lines([line_bgr], num_beams=num_beams)[0]

    def recognize_lines(
        self, line_crops: list[np.ndarray], num_beams: int = 5, batch_size: int = 1
    ) -> list[tuple[str, float]]:
        """Recognize line crops, batch_size at a time.

        batch_size defaults to 1 because batching measurably hurts here:
        on one page, 113 s at 1, 143 s at 2, 155 s at 4, 224 s at 8. Beam
        search generates until the longest line in the batch finishes, so
        every short line pays for the longest one it travels with, and that
        costs more than the per-call overhead batching was meant to save.
        The batching path is kept for the progress reporting it enables.
        """
        results: list[tuple[str, float]] = []
        for start in range(0, len(line_crops), batch_size):
            chunk = line_crops[start:start + batch_size]
            try:
                results.extend(self._recognize_batch(chunk, num_beams))
            except torch.cuda.OutOfMemoryError:
                # A batch of wide crops can exceed a small card; fall back to
                # one at a time rather than failing the whole page.
                torch.cuda.empty_cache()
                for crop in chunk:
                    results.extend(self._recognize_batch([crop], num_beams))
        return results

    def _recognize_batch(
        self, line_crops: list[np.ndarray], num_beams: int
    ) -> list[tuple[str, float]]:
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in line_crops]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

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

        all_texts = [t.strip() for t in self.processor.batch_decode(out.sequences, skip_special_tokens=True)]
        # Per-token probability of each returned sequence. With beam search
        # the raw step scores are indexed by beam slot, not by the winning
        # sequence, so compute_transition_scores resolves that (it also
        # works fine for the num_beams=1 / greedy case).
        beam_indices = getattr(out, "beam_indices", None)
        transition_scores = self.model.compute_transition_scores(
            out.sequences, out.scores, beam_indices, normalize_logits=True
        )

        results = []
        for sample in range(len(line_crops)):
            # generate() returns num_return sequences per input, grouped by
            # input in order: sample 0's beams, then sample 1's, and so on.
            lo = sample * num_return
            texts = all_texts[lo:lo + num_return]

            if self.rescorer and len(texts) > 1:
                scores = [out.sequences_scores[lo + k].item() for k in range(num_return)]
                # sort best-first by the OCR model's own score, since
                # best_candidate() trusts index 0 to reflect its verdict on
                # whether the line is blank -- don't rely on generate()'s
                # output order to already guarantee that.
                order = sorted(range(num_return), key=lambda k: -scores[k])
                candidates = [(texts[k], scores[k]) for k in order]
                best_k = order[self.rescorer.best_candidate(candidates)]
            else:
                best_k = 0

            confidence = 1.0
            probs = torch.exp(transition_scores[lo + best_k])
            probs = probs[torch.isfinite(probs)]
            if probs.numel() > 0:
                confidence = float(probs.mean().item())

            results.append((texts[best_k], confidence))
        return results

    def recognize_page(self, page_bgr: np.ndarray, progress=None, batch_size: int = 1) -> list[RecognizedLine]:
        """progress(done, total) is called as batches complete, so the GUI can
        show movement instead of freezing for the whole page."""
        # A book/notebook spread photo has two independently-flowing text
        # columns; split_pages() detects that and hands back two images
        # (left/right page) instead of one, so segment_lines() below never
        # has to draw a single row projection across both at once -- a
        # single page photo comes back unsplit and this is a no-op.
        segments = []
        for page in split_pages(page_bgr):
            segments.extend(segment_lines(page))

        total = len(segments)
        if progress:
            progress(0, total)

        results = []
        for start in range(0, total, batch_size):
            chunk = segments[start:start + batch_size]
            recognized = self.recognize_lines([crop for _, crop in chunk], batch_size=batch_size)
            for offset, ((bbox, crop), (text, confidence)) in enumerate(zip(chunk, recognized)):
                results.append(
                    RecognizedLine(
                        index=start + offset, bbox=bbox, text=text,
                        confidence=confidence, image=crop,
                    )
                )
            if progress:
                progress(len(results), total)
        return results

    def recognize_file(self, path: str, progress=None) -> list[RecognizedLine]:
        # np.fromfile + imdecode instead of cv2.imread: imread cannot handle
        # non-ASCII paths on Windows, and this project lives under a Cyrillic
        # directory name.
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {path}")
        return self.recognize_page(image, progress=progress)
