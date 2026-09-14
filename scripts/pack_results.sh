#!/usr/bin/env bash
# Everything worth bringing home, in one archive. Safe to call at any moment:
# a rental that ends early still leaves the newest checkpoint behind.
cd ~/ocr || exit 1
out=checkpoints/qwen3vl-hwr200-full
items=(logs stages holdout_pages.jsonl bench/eval_base.jsonl bench/eval_adapter.jsonl
       bench/enhance.jsonl bench/enhance_example.jpg bench/compare_texts.txt)
if [ -f "$out/adapter_model.safetensors" ]; then
  items+=("$out/adapter_model.safetensors" "$out/adapter_config.json")
else
  last=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n 1)
  if [ -n "$last" ]; then
    items+=("$last/adapter_model.safetensors" "$last/adapter_config.json" "$last/trainer_state.json")
  fi
fi
existing=()
for i in "${items[@]}"; do [ -e "$i" ] && existing+=("$i"); done
tar czf results.tar.gz "${existing[@]}"
echo "УПАКОВАНО: ${existing[*]}"
ls -la results.tar.gz
