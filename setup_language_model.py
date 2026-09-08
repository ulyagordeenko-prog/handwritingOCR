"""
Download the Russian language model the recognizer uses to pick between
readings of a line.

The published checkpoint only ships pytorch_model.bin, and recent
transformers refuses to torch.load that on older torch builds, so the
weights are converted to safetensors here once at install time.

Optional: the app runs without this, just slightly less accurately.
"""
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

MODEL_ID = "ai-forever/rugpt3small_based_on_gpt2"
TARGET_DIR = os.path.join("models", "rugpt3small")


def main() -> int:
    if os.path.exists(os.path.join(TARGET_DIR, "model.safetensors")):
        print("Языковая модель уже установлена.")
        return 0

    print(f"Скачиваю {MODEL_ID} (529 МБ)...", flush=True)
    source = snapshot_download(
        MODEL_ID, allow_patterns=["*.json", "*.txt", "pytorch_model.bin"]
    )

    print("Конвертирую в безопасный формат...", flush=True)
    state = torch.load(
        os.path.join(source, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    state = {k: v.contiguous() for k, v in state.items()}
    # GPT-2 ties lm_head to the token embedding; safetensors refuses to
    # store the same storage under two names, and transformers re-ties it
    # on load anyway
    state.pop("lm_head.weight", None)

    os.makedirs(TARGET_DIR, exist_ok=True)
    save_file(state, os.path.join(TARGET_DIR, "model.safetensors"))
    for name in os.listdir(source):
        if name != "pytorch_model.bin":
            shutil.copy(os.path.join(source, name), TARGET_DIR)

    print(f"Готово: {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
