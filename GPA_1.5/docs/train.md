# GPA v1.5 Native Train

> **TL;DR** Use native train when you want to fine-tune or continue training GPA v1.5 with the Hugging Face checkpoint directly. The entrypoint is `GPA-v1.5/train.py`, and if your assets already live in the recommended sibling layout, you can usually start without editing hard-coded paths.

Native train is the simplest way to answer questions like these:

- Does the checkpoint still support mixed ASR and TTS supervision?
- Can I fine-tune with the original JSONL-style data format?
- Can I run a small local smoke check before moving to a larger GPU machine?

If your real target is inference validation, start with `GPA-v1.5/docs/infer.md` instead.

## 📥 Download Index

Before training, download the GPA v1.5 assets from Hugging Face and keep the checkpoint contents together.

<div align="center">

| Asset | Recommended Local Path | Download |
| :--- | :--- | :---: |
| **🤗 GPA-v1.5 Hugging Face checkpoint** | `GPA-v1.5-HF/GPA-v1.5` | **[Download →](https://huggingface.co/AutoArk-AI/GPA-v1.5)** |
| **🎙️ Spark tokenizer assets** | `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model` | Included in the checkpoint |

</div>

> **💡 Tip**: With this layout, `train.py` and `train.sh` can auto-discover both the model directory and the Spark tokenizer directory.

## 📦 Expected Asset Layout

The recommended local layout is:

```text
GPA-v1.5/
GPA-v1.5-HF/
  GPA-v1.5/
    config.json
    model.safetensors
    tokenizer.json
    processing_arkasr.py
    modeling_arkasr.py
    modeling_audio.py
    spark_tokenizer_model/
    ...
```

- `GPA-v1.5-HF/GPA-v1.5` stores the main Hugging Face model assets.
- `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model` stores the Spark tokenizer assets required when TTS samples need reference-voice tokenization.

With this layout, the training CLI can auto-discover:

- the Hugging Face model directory
- the Spark tokenizer directory

If your assets live elsewhere, pass explicit CLI flags or export environment variables.

## 📊 Training Data Format

Training data is JSONL. Each record should declare the task and the fields needed for that task.

Example mixed sample:

```json
{
  "task": "tts",
  "text": "Hello from GPA v1.5.",
  "audio": "/path/to/target.wav",
  "pair_audio": "/path/to/reference.wav",
  "pair_global_ids": [0, 1, 2, 3],
  "audio_semantic_ids": [10, 11, 12, 13]
}
```

Common field usage:

- ASR samples use `audio`, `text`, `begin_time`, and `end_time`
- TTS samples use `text`, `pair_audio`, `pair_global_ids`, and `audio_semantic_ids`

The training loader also accepts the task aliases commonly seen in older GPA data:

- `stt` is treated as ASR
- `tts-a`, `tts_a`, `tts-b`, and `tts_b` are treated as TTS

If audio paths inside the JSONL file are relative paths, they are resolved relative to the JSONL file location.

## 🧰 Environment Notes


> **💡Recommended: create a dedicated Python venv for GPA v1.5 training.**
Do not reuse your system Python or an environment shared with inference or other projects. Training dependencies should live in their own isolated environment.

**Create and activate the training venv:**

```bash
python3 -m venv gpa15-venv
source gpa15-venv/bin/activate
pip install --upgrade pip
pip install -r GPA-v1.5/requirements.train.txt
```

If you prefer conda, keep the environment dedicated to training and install the same requirement file inside it:

```bash
conda create -n gpa15 python=3.10
conda activate gpa15
pip install --upgrade pip
pip install -r GPA-v1.5/requirements.train.txt
```

## Inspect Default Paths

Before launching training, you can inspect what the CLI will use by default:

```bash
python GPA-v1.5/train.py --print-default-paths
```

This prints:

- workspace root
- inferred model directory
- inferred Spark tokenizer directory
- default Hugging Face datasets cache directory
- whether an upstream sample dataset was auto-detected

## 🚀 Quick Start

### Option 1: Wrapper script

The shell wrapper is the quickest path:

```bash
MODEL=/path/to/GPA-v1.5 \
AUDIO_TOKENIZER_PATH=/path/to/spark_tokenizer_model \
DATA=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/output_checkpoint \
bash GPA-v1.5/train.sh
```

Important behavior of `train.sh`:

- it auto-discovers the local model and Spark tokenizer paths when possible
- it falls back to the upstream small sample dataset if that sibling checkout is present
- it uses plain Python when `deepspeed` is not available
- it can switch to DeepSpeed by setting `TRAIN_LAUNCHER=deepspeed`

If you point the wrapper at a raw TTS dataset that does not already contain `pair_global_ids` and `audio_semantic_ids`, add this flag:

```bash
--tts_missing_token_policy fallback_to_tokenizer
```

For mixed-task JSONL training, `remove_unused_columns` is disabled by default in the GPA v1.5 training arguments, so the dataset fields needed by the custom collator are preserved automatically.

### Option 2: Direct Python

If you want full control, call the Python entry directly:

```bash
python GPA-v1.5/train.py \
  --model_name_or_path /path/to/GPA-v1.5 \
  --audio_tokenizer_path /path/to/spark_tokenizer_model \
  --data_path /path/to/train.jsonl \
  --output_dir /path/to/output_checkpoint \
  --do_train \
  --per_device_train_batch_size 1 \
  --tts_missing_token_policy fallback_to_tokenizer \
  --max_steps 1
```

### Option 3: DeepSpeed

When the machine has the required dependencies and a suitable GPU setup, use DeepSpeed through the wrapper:

```bash
TRAIN_LAUNCHER=deepspeed \
DATA=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/output_checkpoint \
bash GPA-v1.5/train.sh
```

By default the wrapper uses `GPA-v1.5/configs/ds_config_zero2.json`.

## 🪄 Shell Wrapper

If you prefer a small wrapper script, use:

```bash
bash GPA-v1.5/train.sh --help
```

The wrapper injects the expected sibling paths automatically:

- `GPA-v1.5-HF/GPA-v1.5`
- `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model`

You can still override them if needed:

```bash
export MODEL=/absolute/path/to/GPA-v1.5
export AUDIO_TOKENIZER_PATH=/absolute/path/to/spark_tokenizer_model
```

## ⚙️ Commonly Used Arguments

The most important training arguments are:

- `--model_name_or_path`
- `--audio_tokenizer_path`
- `--data_path`
- `--eval_data_path`
- `--output_dir`
- `--use_lora`
- `--q_lora`
- `--max_audio_seconds`
- `--tts_max_semantic_tokens`
- `--tts_missing_token_policy`
- `--hf_cache_dir`

Run `python GPA-v1.5/train.py --help` to inspect the full argument surface.

## 🔎 Recommended Smoke Validation

If the current machine can carry a tiny training step, the smallest practical validation is:

```bash
TRAIN_LAUNCHER=python \
DATA=/path/to/merged_shuffled_train.jsonl \
OUTPUT_DIR=tmp_docs/train_smoke \
bash GPA-v1.5/train.sh \
  --per_device_train_batch_size 1 \
  --max_steps 1 \
  --logging_steps 1 \
  --report_to none \
  --dataloader_num_workers 0 \
  --dataloader_persistent_workers False \
  --dataloader_pin_memory False \
  --tts_missing_token_policy fallback_to_tokenizer
```

This smoke path is intentionally conservative:

- plain Python instead of DeepSpeed
- one training step only
- tiny batch size

## 🔫 Fast Troubleshooting

### 1. Model directory cannot be found

Either place the checkpoint in the recommended sibling layout or pass `--model_name_or_path` explicitly.

### 2. Spark tokenizer directory cannot be found

Check this path first:

```text
GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model
```

If your assets live elsewhere, pass `--audio_tokenizer_path` or set `ARK_AUDIO_TOKENIZER_MODEL_DIR`.

### 3. `datasets` or `accelerate` is missing

Activate the dedicated training venv, then reinstall the training dependencies:

```bash
source gpa15-venv/bin/activate
pip install -r GPA-v1.5/requirements.train.txt
```

### 4. DeepSpeed is missing on a local workstation

Use `TRAIN_LAUNCHER=python` first. This is the intended fallback for machines that are not provisioned for multi-GPU training.

### 5. A sample has missing TTS token fields

If a TTS sample does not already include `pair_global_ids` and `audio_semantic_ids`, pass:

```bash
--tts_missing_token_policy fallback_to_tokenizer
```

This lets the training pipeline derive those tokens from `pair_audio`, `ref_audio`, or `audio`.

### 6. Relative audio paths inside JSONL do not resolve

Relative media paths are resolved against the directory of the JSONL file. If you move the JSONL file without its companion media directory, update the paths or move the dataset together.

### 7. `tokenizer.json` is only a Git LFS pointer

If you see errors like `expected value at line 1 column 1` while loading the tokenizer, check the size and content of `tokenizer.json`.

If the file looks like a short Git LFS pointer instead of real JSON, materialize the real asset first, then rerun training. The TTS-specific processor path requires a real fast-tokenizer backend.

The training CLI can fall back to a slow tokenizer for some text-only steps, but full mixed ASR + TTS training still needs the actual tokenizer asset.

### 8. `librosa` fails with a `numba` and `numpy` compatibility error

If audio loading fails with an error like `Numba needs NumPy 2.3 or less`, align the environment to a compatible NumPy release before retrying.

For example, in the dedicated training venv:

```bash
source gpa15-venv/bin/activate
pip install "numpy==1.26.4"
```
