#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a local path or Hugging Face repo id.}"

TASK="${TASK:-NFCorpus}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-mteb_results}"

args=(
  --model-name-or-path "$MODEL_NAME_OR_PATH"
  --task "$TASK"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "${LORA_DIR:-}" ]]; then
  args+=(--lora-dir "$LORA_DIR")
fi

if [[ -n "${CACHE_DIR:-}" ]]; then
  args+=(--cache-dir "$CACHE_DIR")
fi

python "$(dirname "$0")/eval_mteb.py" "${args[@]}"
