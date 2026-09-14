#!/usr/bin/env bash
# The second fine-tune, from a bare rented box to a packed result, unattended.
#
# Every stage leaves a marker in stages/ when it succeeds and is skipped on a
# rerun, so after a dropped link or a reboot the same command carries on where
# it stopped. Training resumes from its last checkpoint by itself.
#
#   setsid nohup bash scripts/run_all.sh >> logs/run_all.log 2>&1 &
set -uo pipefail
cd ~/ocr || exit 1
echo $$ > run_all.pid
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=src PYTHONUNBUFFERED=1
# The weights alone are 17 GB; the hub's default timeouts give up on a slow
# mirror halfway through.
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
PY=.venv/bin/python
ADAPTER=checkpoints/qwen3vl-hwr200-full
mkdir -p stages logs bench checkpoints

stage() {
  local name=$1; shift
  if [ -f "stages/$name" ]; then echo "== $name: уже сделано"; return 0; fi
  echo "== $name: начало $(date '+%d.%m %H:%M')"
  if "$@" >> "logs/$name.log" 2>&1; then
    touch "stages/$name"
    echo "== $name: готово $(date '+%d.%m %H:%M')"
  else
    echo "== $name: ОШИБКА $(date '+%d.%m %H:%M') -- logs/$name.log:"
    tail -c 3000 "logs/$name.log" | tr '\r' '\n' | tail -n 25
    exit 1
  fi
}

install() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  [ -d .venv ] || uv venv --python 3.11 .venv
  # cu128 is the oldest CUDA build that runs on an RTX 50 card, and it still
  # runs on a 3090, a 4090 or an A100 -- whichever the rental turns out to be.
  uv pip install --python "$PY" torch "torchvision>=0.20" \
      --index-url https://download.pytorch.org/whl/cu128 || return 1
  # The versions the app itself runs, so the adapter trained here loads there.
  uv pip install --python "$PY" "transformers==5.15.1" "peft==0.20.0" "accelerate==1.14.0" \
      "bitsandbytes==0.50.2" "huggingface_hub>=1.28" opencv-python-headless scipy \
      rapidfuzz pillow safetensors
}

check_gpu() {
  "$PY" - <<'EOF'
import torch, torchvision, transformers, peft, cv2
import bitsandbytes as bnb
props = torch.cuda.get_device_properties(0)
print("видеокарта:", props.name, "| память", round(props.total_memory / 1024**3, 1), "ГБ")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
# is_available() says yes on a card the build cannot actually run; an
# operation does not lie, and the 4-bit kernel is the one training leans on
x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
print("4 бита работают:", tuple(layer(x).shape))
EOF
}

download() {
  "$PY" - <<'EOF' || return 1
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-8B-Instruct")
print("МОДЕЛЬ_НА_МЕСТЕ")
EOF
  "$PY" scripts/fetch_hwr200.py
}

smoke() {
  # The heaviest pages, at the resolution the first run trained on. A card
  # that cannot hold them gets a smaller page now rather than a crash at hour
  # six; the resolution that fits is kept for the real run. Any failure other
  # than running out of memory is a bug, and stops here.
  local pixels code
  for pixels in $((1280*32*32)) $((1024*32*32)) $((768*32*32)); do
    echo "пробую max_pixels=$pixels"
    "$PY" scripts/train_qwen.py --smoke 12 --max-pixels "$pixels"
    code=$?
    if [ "$code" -eq 0 ]; then echo "$pixels" > stages/max_pixels; return 0; fi
    if [ "$code" -ne 3 ] && [ "$code" -ne 4 ]; then return "$code"; fi
  done
  return 1
}

train() {
  "$PY" scripts/train_qwen.py --max-pixels "$(cat stages/max_pixels)" --out "$ADAPTER"
}

stage install install
stage check_gpu check_gpu
stage download download
stage smoke smoke
stage eval_base "$PY" scripts/eval_adapter.py --label base
stage enhance "$PY" scripts/bench_enhance.py
stage train train
stage eval_adapter "$PY" scripts/eval_adapter.py --label adapter --adapter "$ADAPTER"
stage compare "$PY" scripts/eval_adapter.py --compare
bash scripts/pack_results.sh
echo "ВСЁ_ГОТОВО $(date '+%d.%m %H:%M')"
