#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREFERRED_MODEL_PATH="${SCRIPT_DIR}/../GPA-v1.5-HF/GPA-v1.5"
LEGACY_MODEL_PATH="${SCRIPT_DIR}/../GPA-v1.5-HF/GPA_v1.5"
PREFERRED_AUDIO_TOKENIZER_PATH="${PREFERRED_MODEL_PATH}/spark_tokenizer_model"
LEGACY_AUDIO_TOKENIZER_PATH="${LEGACY_MODEL_PATH}/spark_tokenizer_model"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${PREFERRED_MODEL_PATH}" || ! -d "${LEGACY_MODEL_PATH}" ]]; then
    MODEL_PATH="${PREFERRED_MODEL_PATH}"
  else
    MODEL_PATH="${LEGACY_MODEL_PATH}"
  fi
fi

if [[ -z "${AUDIO_TOKENIZER_PATH:-}" ]]; then
  if [[ -d "${PREFERRED_AUDIO_TOKENIZER_PATH}" || ! -d "${LEGACY_AUDIO_TOKENIZER_PATH}" ]]; then
    AUDIO_TOKENIZER_PATH="${PREFERRED_AUDIO_TOKENIZER_PATH}"
  else
    AUDIO_TOKENIZER_PATH="${LEGACY_AUDIO_TOKENIZER_PATH}"
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import soundfile  # noqa: F401
import torch  # noqa: F401
import torchaudio  # noqa: F401
import transformers  # noqa: F401
from tokenizers import Regex  # noqa: F401
PY
then
  echo "[error] ${PYTHON_BIN} is missing native inference dependencies. Activate the prepared environment first." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
    --model-path "${MODEL_PATH}" \
    --audio-tokenizer-path "${AUDIO_TOKENIZER_PATH}" \
    --help
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
  --model-path "${MODEL_PATH}" \
  --audio-tokenizer-path "${AUDIO_TOKENIZER_PATH}" \
  "$@"