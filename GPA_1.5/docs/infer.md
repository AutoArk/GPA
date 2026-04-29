# GPA v1.5 Native Infer

> **TL;DR** Use native infer when you want the most direct GPA v1.5 behavior baseline for ASR and TTS. The entrypoint is `GPA-v1.5/infer.py`, and if your assets already live in the recommended sibling layout, you can usually run it without any path editing.

Native infer is the simplest way to answer questions like these:

- Does the Hugging Face checkpoint load correctly?
- Does the processor really inject audio into ASR?
- Can the model still generate semantic speech tokens for TTS?
- What does the direct PyTorch path do before any deployment-specific wrapping?

## 📥 Download Index

Before running inference, download the GPA v1.5 assets from Hugging Face and keep the checkpoint contents together.

<div align="center">

| Asset | Recommended Local Path | Download |
| :--- | :--- | :---: |
| **🤗 GPA-v1.5 Hugging Face checkpoint** | `GPA-v1.5-HF/GPA-v1.5` | **[Download →](https://huggingface.co/AutoArk-AI/GPA-v1.5)** |
| **🎙️ Spark tokenizer assets** | `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model` | Included in the checkpoint |

</div>

> **💡 Tip**: With this layout, `infer.py` and `infer.sh` can auto-discover both the model directory and the Spark tokenizer directory.

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
- `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model` stores the Spark tokenizer assets required by TTS voice conditioning.

The CLI auto-discovers both locations inside the same model directory tree.

## 🧰 Environment Notes


> **💡Recommended: create a new, isolated Python environment for GPA v1.5.**
Using your system Python or an existing environment shared with other projects can cause package conflicts.

**venv example:**

```bash
python3 -m venv gpa15-venv
source gpa15-venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio transformers tokenizers soundfile librosa numpy
```

Or use conda if you prefer:

```bash
conda create -n gpa15 python=3.10
conda activate gpa15
pip install torch torchaudio transformers tokenizers soundfile librosa numpy
```

**Required dependencies:**

- torch
- torchaudio
- transformers
- tokenizers
- soundfile
- librosa
- numpy

## 🔎 Check What The CLI Will Use

Before running inference, you can inspect the auto-discovered paths:

```bash
python GPA-v1.5/infer.py --print-default-paths
```

This is useful when you want to confirm:

- which workspace root was inferred
- which HF model directory will be used
- which Spark tokenizer directory will be used
- whether an environment variable override is already active

## 🚀 Quick Start

### 🎧 ASR

Run a direct speech recognition pass:

```bash
python GPA-v1.5/infer.py \
  --task asr \
  --asr-audio GPA-v1.5/samples/sample.mp3
```

Useful knobs:

- `--begin-time` and `--end-time` to crop the audio region
- `--asr-max-new-tokens` to limit decode length
- `--device cpu|mps|cuda` to choose the runtime device

### 🎙️ TTS

Run a direct speech synthesis pass with a reference clip:

```bash
python GPA-v1.5/infer.py \
  --task tts \
  --ref-audio GPA-v1.5/samples/sample.mp3 \
  --tts-text "Hello, this is a native GPA speech synthesis check." \
  --tts-out-wav tmp_docs/native_infer_smoke/tts.wav
```

This TTS path does four things in order:

1. extracts global speaker tokens from the reference audio
2. builds the TTS prompt with those tokens
3. generates semantic speech tokens from the main model
4. reconstructs waveform audio through SparkDeTokenizer

## 🪄 Shell Wrapper

If you prefer a small wrapper script, use:

```bash
bash GPA-v1.5/infer.sh --help
```

The wrapper injects the expected sibling paths automatically:

- `GPA-v1.5-HF/GPA-v1.5`
- `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model`

You can still override them if needed:

```bash
export MODEL_PATH=/absolute/path/to/GPA-v1.5
export AUDIO_TOKENIZER_PATH=/absolute/path/to/spark_tokenizer_model
```

## ⚙️ Commonly Used Arguments

The most important CLI options are:

- `--task`: choose `asr` or `tts` (`vc` coming soon)
- `--device`: choose `cpu`, `mps`, or `cuda`
- `--model-path`: override the HF model directory
- `--audio-tokenizer-path`: override the Spark tokenizer directory
- `--tts-out-wav`: choose the saved waveform path
- `--dump-tts-raw`: write the raw TTS generation beside the output wav

For compatibility with the original script, underscore-style arguments are still accepted, including:

- `--model_path`
- `--audio_tokenizer_path`
- `--tts_text`

## 🔫 Fast Troubleshooting

### 1. The model directory cannot be found

If the CLI says the model directory is missing, either:

- place the model assets in `GPA-v1.5-HF/GPA-v1.5`
- pass `--model-path`
- or set `ARK_AUDIO_HF_MODEL_DIR`

### 2. The Spark tokenizer directory cannot be found

If TTS fails before generation starts, first check this path:

```text
GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model
```

If your assets live elsewhere, use `--audio-tokenizer-path` or set `ARK_AUDIO_TOKENIZER_MODEL_DIR`.

### 3. ASR says audio was not encoded

That usually means the processor and model assets are not matched correctly, or the asset set is incomplete.

### 4. TTS fails to parse semantic tokens

If no semantic tokens are found:

- confirm the reference audio is readable
- confirm the Spark tokenizer assets belong to the same model family
- try a shorter synthesis text first
- use `--dump-tts-raw` and inspect the raw generation text

### 5. Device behavior differs across machines

On macOS, `mps` may not always be the best choice for debugging. If you want the most predictable path, force CPU:

```bash
python GPA-v1.5/infer.py \
  --task asr \
  --asr-audio GPA-v1.5/samples/sample.mp3 \
  --device cpu
```

The CLI chooses the attention backend conservatively:

- `flash_attention_2` on CUDA when available
- `sdpa` on MPS
- `eager` on CPU
