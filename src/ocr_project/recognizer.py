"""
Page -> lines -> text recognition using TrOCR-ru + our LoRA adapter.

Exposes a single Recognizer class that the GUI drives: it loads the model
once, then turns a page image into a list of RecognizedLine records that
carry both the transcription and a per-line confidence, so the UI can flag
places the model was unsure about.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .rescorer import LanguageRescorer
from .segmentation import segment_lines, split_pages

log = logging.getLogger(__name__)

BASE_MODEL_ID = "raxtemur/trocr-base-ru"
DEFAULT_ADAPTER = "checkpoints/trocr-lora-v5"

# A single line stuck this long is not read, it is skipped -- much shorter
# than page_reader.py's whole-page GENERATION_TIMEOUT_S, since one line's
# beam search normally takes single-digit seconds (see recognize_lines'
# own docstring: ~113 s for a full page at batch_size=1). Skipping just the
# stuck line and moving on is only possible here because each line is
# already its own generate() call in a loop -- the whole-page model reads
# the entire page as one continuous stream with no such boundary to stop
# at and resume from, which is why it gets a page-wide timeout instead.
LINE_TIMEOUT_S = 20.0


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
            f"это работает, но медленнее. Чтение страницы целиком в таком режиме "
            f"недоступно.\n\n"
            f"Чтобы это исправить, закройте приложение и запустите файл "
            f"fix_gpu.bat — он подберёт подходящую сборку.\n\n"
            f"Подробность: {exc}"
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
        started = time.monotonic()
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
                stopping_criteria=_line_stopping_criteria(started),
            )
        # A stuck line, not a slow one -- see LINE_TIMEOUT_S. Each line is
        # its own generate() call already, so unlike the whole-page model
        # this can just skip it (confidence 0, tinted for a second look) and
        # let the loop in recognize_lines move on to the next one, rather
        # than losing the rest of the page the way an unbounded stuck line
        # otherwise would.
        stuck = time.monotonic() - started >= LINE_TIMEOUT_S
        if stuck:
            log.warning("_recognize_batch: line stuck past %.0fs, skipped", LINE_TIMEOUT_S)

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
            if stuck:
                # Overrides whatever the truncated beam's own score says --
                # a forced cut mid-search is not evidence of confidence
                # either way, and this line needs a second look regardless.
                confidence = 0.0

            results.append((texts[best_k], confidence))
        return results

    def recognize_page(self, page_bgr: np.ndarray, progress=None, batch_size: int = 1,
                        num_beams: int = 5, cancel_event=None, on_line=None) -> list[RecognizedLine]:
        """progress(done, total) is called as batches complete, so the GUI can
        show movement instead of freezing for the whole page.

        on_line(RecognizedLine), if given, is called for each line right as
        it is recognized -- unlike the whole-page model, every line is
        already its own step in this loop, so there is no need to stream
        partial output mid-generation the way _generate_streamed does for
        that one; the line itself, done or not, is the natural unit here.

        num_beams is exposed (rather than fixed at recognize_lines' own
        default) for callers that want a cheaper, rougher pass -- the
        whole-page cross-check in app.py only needs an approximate second
        opinion to fuzzy-match against, not this model's best possible
        reading, and beam search is the slow part of it.

        cancel_event, if given, is checked once per batch: each line's own
        beam search is short enough that stopping between batches, rather
        than needing a StoppingCriteria mid-line the way the much slower
        whole-page model does, is already responsive."""
        # A book/notebook spread photo has two independently-flowing text
        # columns; split_pages() detects that and hands back two images
        # (left/right page) instead of one, so segment_lines() below never
        # has to draw a single row projection across both at once -- a
        # single page photo comes back unsplit and this is a no-op.
        started = time.monotonic()
        segments = []
        for page in split_pages(page_bgr):
            segments.extend(segment_lines(page))

        total = len(segments)
        if progress:
            progress(0, total)

        results = []
        cancelled = False
        for start in range(0, total, batch_size):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            chunk = segments[start:start + batch_size]
            recognized = self.recognize_lines([crop for _, crop in chunk], num_beams=num_beams,
                                              batch_size=batch_size)
            for offset, ((bbox, crop), (text, confidence)) in enumerate(zip(chunk, recognized)):
                line = RecognizedLine(
                    index=start + offset, bbox=bbox, text=text,
                    confidence=confidence, image=crop,
                )
                results.append(line)
                if on_line:
                    on_line(line)
            if progress:
                progress(len(results), total)
        # Not the handwriting's own text here either -- see page_reader.py's
        # module docstring for why: lengths and confidence only.
        log.info("recognize_page: %.1fs, %d/%d lines, num_beams=%d%s",
                 time.monotonic() - started, len(results), total, num_beams,
                 " [cancelled]" if cancelled else "")
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


def _line_stopping_criteria(started: float):
    """Ends one line's beam search once LINE_TIMEOUT_S has passed since
    `started` -- see _recognize_batch. Same mechanism as page_reader.py's
    _stopping_criteria, kept separate rather than shared: that one also
    watches a cancel_event, this one only ever watches the clock."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Stop(StoppingCriteria):
        def __call__(self, *_args, **_kwargs) -> bool:
            return time.monotonic() - started >= LINE_TIMEOUT_S

    return StoppingCriteriaList([_Stop()])
