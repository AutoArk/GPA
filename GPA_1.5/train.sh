#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-auto}"
MASTER_PORT="${MASTER_PORT:-29502}"

PREFERRED_MODEL_PATH="${SCRIPT_DIR}/../GPA-v1.5-HF/GPA-v1.5"
LEGACY_MODEL_PATH="${SCRIPT_DIR}/../GPA-v1.5-HF/GPA_v1.5"
PREFERRED_AUDIO_TOKENIZER_PATH="${PREFERRED_MODEL_PATH}/spark_tokenizer_model"
LEGACY_AUDIO_TOKENIZER_PATH="${LEGACY_MODEL_PATH}/spark_tokenizer_model"
DEFAULT_SAMPLE_DATA="${SCRIPT_DIR}/../../GPA/scripts/train/merged_shuffled_train.jsonl"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/outputs/train-checkpoint"
DEFAULT_DS_CONFIG="${SCRIPT_DIR}/configs/ds_config_zero2.json"
DEFAULT_HOSTFILE="${SCRIPT_DIR}/configs/hostfile.example"

if [[ -z "${MODEL:-}" ]]; then
  if [[ -d "${PREFERRED_MODEL_PATH}" || ! -d "${LEGACY_MODEL_PATH}" ]]; then
    MODEL="${PREFERRED_MODEL_PATH}"
  else
    MODEL="${LEGACY_MODEL_PATH}"
  fi
fi

if [[ -z "${AUDIO_TOKENIZER_PATH:-}" ]]; then
  if [[ -d "${PREFERRED_AUDIO_TOKENIZER_PATH}" || ! -d "${LEGACY_AUDIO_TOKENIZER_PATH}" ]]; then
    AUDIO_TOKENIZER_PATH="${PREFERRED_AUDIO_TOKENIZER_PATH}"
  else
    AUDIO_TOKENIZER_PATH="${LEGACY_AUDIO_TOKENIZER_PATH}"
  fi
fi

if [[ -z "${DATA:-}" && -f "${DEFAULT_SAMPLE_DATA}" ]]; then
  DATA="${DEFAULT_SAMPLE_DATA}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-${DEFAULT_DS_CONFIG}}"
HOSTFILE_PATH="${HOSTFILE_PATH:-${DEFAULT_HOSTFILE}}"

USE_LORA="${USE_LORA:-False}"
Q_LORA="${Q_LORA:-False}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-True}"
PIN_MEMORY="${PIN_MEMORY:-True}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRIPT_DIR}/.cache/hf_datasets}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.cache/hf_home}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${OUTPUT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

show_help_only=0
for arg in "$@"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" || "${arg}" == "--print-default-paths" ]]; then
    show_help_only=1
    break
  fi
done

if [[ $# -eq 0 ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" --help
fi

if [[ "${show_help_only}" == "1" ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" "$@"
fi

if [[ -z "${DATA:-}" ]]; then
  echo "[error] DATA is not set and no upstream sample dataset was found." >&2
  echo "[error] Set DATA=/path/to/train.jsonl before running training." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import datasets  # noqa: F401
import soundfile  # noqa: F401
import torch  # noqa: F401
import torchaudio  # noqa: F401
import transformers  # noqa: F401
from accelerate import Accelerator  # noqa: F401
PY
then
  echo "[error] ${PYTHON_BIN} is missing training dependencies. Install GPA-v1.5/requirements.train.txt or activate the prepared environment first." >&2
  exit 1
fi

launcher_choice="${TRAIN_LAUNCHER}"
if [[ "${launcher_choice}" == "auto" ]]; then
  if command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1; then
    launcher_choice="deepspeed"
  else
    launcher_choice="python"
  fi
fi

common_args=(
  --model_name_or_path "${MODEL}"
  --audio_tokenizer_path "${AUDIO_TOKENIZER_PATH}"
  --data_path "${DATA}"
  --output_dir "${OUTPUT_DIR}"
  --bf16 True
  --num_train_epochs 3
  --per_device_train_batch_size 10
  --gradient_accumulation_steps 1
  --save_strategy steps
  --save_steps 1000
  --save_total_limit 3
  --learning_rate 3e-6
  --weight_decay 0.005
  --adam_beta2 0.95
  --do_train
  --dataloader_drop_last True
  --warmup_ratio 0.01
  --lr_scheduler_type cosine
  --logging_steps 1
  --report_to tensorboard
  --model_max_length 1000
  --remove_unused_columns False
  --ddp_find_unused_parameters True
  --gradient_checkpointing False
  --use_lora "${USE_LORA}"
  --q_lora "${Q_LORA}"
  --dataloader_num_workers "${NUM_WORKERS}"
  --dataloader_pin_memory "${PIN_MEMORY}"
)

if [[ "${NUM_WORKERS}" != "0" ]]; then
  common_args+=(
    --dataloader_persistent_workers "${PERSISTENT_WORKERS}"
    --dataloader_prefetch_factor "${PREFETCH_FACTOR}"
  )
fi

if [[ -n "${EVAL_DATA:-}" ]]; then
  common_args+=(
    --eval_data_path "${EVAL_DATA}"
    --per_device_eval_batch_size 10
    --eval_strategy steps
    --eval_steps 1000
  )
fi

if [[ "${launcher_choice}" == "deepspeed" ]]; then
  if ! command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1; then
    echo "[error] deepspeed launcher not found: ${DEEPSPEED_BIN}" >&2
    exit 1
  fi

  launcher=("${DEEPSPEED_BIN}" --master_port "${MASTER_PORT}")
  if [[ -f "${HOSTFILE_PATH}" ]]; then
    launcher+=(--hostfile "${HOSTFILE_PATH}")
  fi

  exec "${launcher[@]}" "${SCRIPT_DIR}/train.py" \
    --deepspeed "${DS_CONFIG_PATH}" \
    "${common_args[@]}" \
    "$@"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" \
  "${common_args[@]}" \
  "$@"