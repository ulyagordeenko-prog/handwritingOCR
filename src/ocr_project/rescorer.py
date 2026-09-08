"""
A small Russian language model used to pick between OCR candidates, not to
invent text. TrOCR's beam search already proposes several visually-plausible
readings for a line; this only re-ranks those existing candidates by how
much they read like real Russian, using their average per-token log
probability under ruGPT3-small. It never generates or edits text on its
own, so it can't introduce a word the OCR model didn't already consider --
important for a document with real personal data, where a "cleaner-sounding"
invention is worse than an odd-looking truth.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LM_DIR = "models/rugpt3small"


class LanguageRescorer:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(LM_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(LM_DIR).to(self.device)
        self.model.eval()

    def score(self, text: str) -> float:
        """Average log-probability per token under the LM. Higher = more
        plausible Russian. Empty text is scored neutrally (0.0) rather than
        penalized -- an actually-blank line shouldn't lose to a hallucinated
        one just because the LM has nothing to judge."""
        text = text.strip()
        if not text:
            return 0.0
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        if ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            out = self.model(ids, labels=ids)
        return -out.loss.item()

    def best_candidate(self, candidates: list[tuple[str, float]], lm_weight: float = 0.5) -> int:
        """candidates: list of (text, ocr_score), assumed already sorted
        best-first by the OCR model's own beam score (index 0 = OCR's own
        top pick). Returns the index of the chosen candidate.

        Whether the line is blank is left entirely to the OCR model: an
        empty string has no real log-probability to compare against real
        text (0.0 is the mathematical ceiling for a log-probability, so a
        naive "neutral" score for it would always beat genuine text and
        make the model go silent on lines it actually read fine). So if
        the OCR model's own top candidate is blank, that's respected as-is
        -- the language model only ever adjudicates between the non-blank
        readings, picking whichever reads most like real Russian.
        """
        if not candidates[0][0]:
            return 0

        best_i, best_combined = None, float("-inf")
        for i, (text, ocr_score) in enumerate(candidates):
            if not text:
                continue
            combined = ocr_score + lm_weight * self.score(text)
            if combined > best_combined:
                best_i, best_combined = i, combined
        return best_i if best_i is not None else 0
