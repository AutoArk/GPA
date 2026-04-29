# GPA v1.5 ONNX Runtime

Welcome to the ONNX runtime codebase for GPA v1.5.

Instead of giving you a black-box executable, this directory contains everything you need to run, inspect, and debug the model locally.

You can spin up a FastAPI server, run local TTS/ASR, track memory usage, register custom voices, and even step through a Torch vs. ONNX comparison.

## 📦 Download the Runtime Assets

Download the ONNX runtime asset bundle from **[Hugging Face](https://huggingface.co/AutoArk-AI/GPA-v1.5-onnx-runtime)**.

> **💡 Recommended layout:** Download the runtime asset bundle from Hugging Face into a sibling path: `GPA-v1.5-HF/GPA-v1.5-onnx-runtime`.

This path is recommended because the runtime code already looks for it automatically. If your local layout matches this path, you do not need any extra configuration.

If you store the assets somewhere else, point the runtime to them explicitly with `ARK_AUDIO_RUNTIME_ASSET_ROOT`.

**Example:**

```bash
export ARK_AUDIO_RUNTIME_ASSET_ROOT=/absolute/path/to/GPA-v1.5-onnx-runtime
```

After exporting that variable, the CLI tools and `service.py` will load assets from your custom location instead of the default sibling path.

### 🔎 Post-Download Structure Check

Before you run inference, make sure the downloaded asset bundle contains these directories:

```text
GPA-v1.5-HF/GPA-v1.5-onnx-runtime/
├── build/
├── genai_fp16_qwen/
├── genai_int4_qwen/
├── model/
└── voice/
    └── spark_tokenizer_model/
```

The most important checks are:

- `model/runtime_manifest.json`
- `model/reference/default_global_tokens.npy`
- `genai_fp16_qwen/model.onnx`
- `genai_int4_qwen/model.onnx`
- `voice/spark_tokenizer_model/config.yaml`
- `voice/spark_tokenizer_model/model.safetensors`

### 🧪 Quick Validation Commands (Optional)

You can sanity-check the local layout with the following commands:

```bash
test -d GPA-v1.5-HF/GPA-v1.5-onnx-runtime/model
test -d GPA-v1.5-HF/GPA-v1.5-onnx-runtime/build
test -d GPA-v1.5-HF/GPA-v1.5-onnx-runtime/voice/spark_tokenizer_model
test -f GPA-v1.5-HF/GPA-v1.5-onnx-runtime/model/runtime_manifest.json
test -f GPA-v1.5-HF/GPA-v1.5-onnx-runtime/voice/spark_tokenizer_model/model.safetensors
```

## 📂 ONNX Codebase Quick Tour

**The Codebase** (`GPA-v1.5/onnx_runtime`) consists of following components:
* `infer_ark_audio_*.py`: Main inference scripts for standard and GenAI-based TTS/ASR.
* `service.py` & `static/`: FastAPI backend and the browser-based UI.
* `voice_registration.py` & `voice_registry.py`: Voice cloning and token management.
* `compare_torch_onnx_tts.py`: A handy tool for step-by-step debugging between Torch and ONNX.
* `build_runtime.py` & `scripts/`: Build orchestration, export, and quantization utilities.
* `outputs/`, `voices/`, `samples/`: Local state directories for generated audio and voice metadata.

---

## 🛠️ Environment Setup

Using a dedicated `venv` environment is recommended.

**Recommended setup:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r GPA-v1.5/onnx_runtime/requirements.runtime.txt
```
---

## 🚀 Quick Start

### 1. **CLI Smoke Tests**
Make sure everything is working by running a quick TTS and ASR pass after the Hugging Face asset bundle has been downloaded and placed in the expected local path.

**Generate Speech (TTS - int4):**
```bash
python GPA-v1.5/onnx_runtime/infer_ark_audio_onnx.py \
  --runtime-root GPA-v1.5-HF/GPA-v1.5-onnx-runtime \
  --task tts \
  --tts_text "Hello, this is a short GPA speech synthesis check." \
  --tts_out_wav tmp_docs/onnx_smoke/readme_tts_int4_int8.wav \
  --main-model-precision int4 \
  --decoder-precision int8
```

By default, the CLI uses `GPA-v1.5-HF/GPA-v1.5-onnx-runtime/model/reference/default_global_tokens.npy` from the runtime asset bundle. If you want to synthesize with a specific registered voice, pass its token file explicitly:

```bash
python GPA-v1.5/onnx_runtime/infer_ark_audio_onnx.py \
  --runtime-root GPA-v1.5-HF/GPA-v1.5-onnx-runtime \
  --task tts \
  --tts_text "Hello, this is a short GPA speech synthesis check." \
  --tts_out_wav tmp_docs/onnx_smoke/readme_tts_registered_voice.wav \
  --voice-global-token GPA-v1.5/onnx_runtime/voices/items/default/global_tokens.npy \ # Replace with the path of .npy file of your registered voice
  --main-model-precision int4 \
  --decoder-precision int8
```

**Transcribe the Audio (ASR - int4):**
```bash
python GPA-v1.5/onnx_runtime/infer_ark_audio_onnx.py \
  --runtime-root GPA-v1.5-HF/GPA-v1.5-onnx-runtime \
  --task asr \
  --asr_audio tmp_docs/onnx_smoke/readme_tts_int4_int8.wav \
  --main-model-precision int4
```

**Got a beefy GPU?** Swap the precisions to `fp16` to test the high-fidelity outputs.

### 2. **Boot up the Local Web Service**
We've included a FastAPI app with a built-in browser UI for easy testing.

```bash
cd GPA-v1.5/onnx_runtime
uvicorn service:app --host 127.0.0.1 --port 8024
```
Open up `http://127.0.0.1:8024/` in your browser. From here, you can generate speech, upload/record audio for ASR, check the default voice, and monitor runtime RSS and peak memory usage.

Similarly, if your asset bundle is not in the default path, export `ARK_AUDIO_RUNTIME_ASSET_ROOT` before starting the service.

---

## 🔌 API Endpoints

The FastAPI service exposes a variety of endpoints, including OpenAI-compatible routes:

* **UI:** `GET /`
* **Observability:** `GET /api/health`, `/api/memory`, `/api/voices`
* **Core Inference:** `POST /api/tts`, `/api/asr`
* **Voice Management:** `POST /api/voices/register-path`, `/api/voices/register-upload`
* **OpenAI Compatible:** `GET /v1/models`, `GET /v1/audio/voices`, `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`

---

## 🎙️ Voice Registration

Before performing voice registration, make sure the following path of assets exists:

- `GPA-v1.5-HF/GPA-v1.5-onnx-runtime/voice/spark_tokenizer_model`

Similarly, if the assets are located at non-default directory, you can export the corresponding path before starting the service:

```bash
export ARK_AUDIO_TOKENIZER_MODEL_DIR=/absolute/path/to/spark_tokenizer_model
```

You can verify the registration path from the UI or with `POST /api/voices/register-path`. A successful registration writes metadata under `GPA-v1.5/onnx_runtime/voices/`.

Each registered voice stores its TTS control tokens under:

- `GPA-v1.5/onnx_runtime/voices/items/<voice_id>/global_tokens.npy`

That file can be passed directly to the CLI with `--voice-global-token`.

Typical workflow:

1. Register a voice from the UI.
2. Check `GPA-v1.5/onnx_runtime/voices/registry.json` to find the saved `voice_id`.
3. Run the CLI with `--voice-global-token GPA-v1.5/onnx_runtime/voices/items/<voice_id>/global_tokens.npy`.

*⚠️ Note: If you remove the bundled tokenizer assets or point the override to a missing directory, the service will still start, and the default bundled voice will still work. However, the voice registration endpoints will return a clean error.*

---

## 🔧 Developer Tooling

We left our dev scripts in the repo so you can reproduce our builds or debug your own forks:

* **Torch vs ONNX debugging:** Inspect graph differences step-by-step.
    ```bash
    python compare_torch_onnx_tts.py --help
    ```
* **Rebuild runtime assets:** Refresh the asset bundle.
    ```bash
    python build_runtime.py --help
    ```
* **Export/Quantization:** Check the `scripts/` folder for all our export, quant, and sweep utilities.

---

## ⚠️ Status & Known Issues

This runtime was recently validated on macOS (testing both `int4` and `fp16` flows for CLI inference, FastAPI startup, and UI functionality). 

**Keep in mind:**
* **UI Sample File:** The browser UI looks for `samples/sample.mp3` by default. If it's missing, the UI will let you know safely.
* **ASR Accuracy:** The ONNX CLI/service smoke tests check for successful execution, not perfect transcription accuracy. Expect minor wording variations. 
* **Voice Reg:** The default packaged layout now includes the tokenizer model assets under the HF bundle. Use `ARK_AUDIO_TOKENIZER_MODEL_DIR` only if you want to replace them.