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

If the reason you're troubleshooting is a `flash_attention_2` import error on a CUDA GPU (see the [Tesla T4 / Turing notes](#-tesla-t4--turing-sm_75-notes) below), you don't need to give up the GPU — pass `--attn-impl sdpa` instead and keep `--device cuda`.

By default (`--attn-impl auto`), the CLI picks:

- `flash_attention_2` whenever `--device` starts with `cuda` — **regardless of GPU generation or whether the `flash-attn` package is even installed**
- `sdpa` on MPS
- `eager` on CPU

## 🖥️ Tesla T4 / Turing (sm_75) Notes

Measured on a real Tesla T4 16GB (driver 550.163.01 / CUDA 12.4, torch 2.6.0+cu124, transformers 5.13.0). These notes apply to any Turing-class GPU (T4, and similarly L4-class cards without `flash-attn` support).

- **`--attn-impl auto` crashes on Turing GPUs.** `inference/model_loader.py::normalize_attn_impl` returns `flash_attention_2` for any `cuda*` device with no check of `torch.cuda.get_device_capability()` and no check that the `flash-attn` package is installed (it isn't listed in `requirements.txt`, and `pyproject.toml` only has it commented out as an optional extra). On T4 (sm_75), the Quick Start commands above fail at model-load time with:
  ```
  ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error:
  the package for FlashAttention2 doesn't seem to be installed.
  ```
  `flash-attn` v2 also requires sm80+ in general, so it would not help on T4 even if installed. **Fix:** explicitly pass `--attn-impl sdpa` — this is already a supported CLI value, no code change needed, and both ASR and TTS complete normally.

- **There is no `--dtype` CLI flag.** `--device` and `--attn-impl` are exposed as CLI options, but dtype is hardcoded in `model_loader.py` (`torch.bfloat16` on CUDA, otherwise the model default). Changing it requires editing `load_text_stack()` directly. On T4, this hardcoded `bf16` is measurably suboptimal with SDPA: PyTorch's `EFFICIENT_ATTENTION` kernel rejects `bfloat16` inputs and silently falls back to the slower `MATH` backend, whereas `fp16` is accepted by `EFFICIENT_ATTENTION` directly. Per-token generation time was within noise between the two in our runs (bf16 ≈36.6–37.2 ms/token vs fp16 ≈34.2–39.4 ms/token), but only `fp16` reaches the faster kernel path on this GPU class.

- **`bitsandbytes` int8 quantization works, but is a real trade-off, not a free win.** Passing `quantization_config=BitsAndBytesConfig(load_in_8bit=True)` to `AutoModelForCausalLM.from_pretrained` (not currently mentioned anywhere in the docs) reduces peak VRAM after load by **~43%** (2317 MB → 1327 MB bf16→int8, reproduced twice), but batch=1 autoregressive decode is **~3.2x slower** (≈36–39 ms/token bf16/fp16 → ≈121–125 ms/token int8, reproduced twice). Whether this trade is worth it depends on whether you're VRAM-constrained or latency-constrained.

- **VRAM headroom on T4 is not a concern for this model.** GPA-v1.5's native infer path uses a 0.6B `ArkAsr` backbone. Full-pipeline peak VRAM (LLM + ASR/TTS + Spark tokenizer) measured ≈4.0–4.2 GB for bf16/fp16 and ≈3.0–3.1 GB for int8 — roughly 74–81% of a T4's 16GB left unused. Unlike larger models, OOM is not a practical risk here on T4-class hardware.
