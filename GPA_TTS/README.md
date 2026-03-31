<div align="center">

# GPA-TTS

### Lightweight Voice-Cloning TTS for Edge Devices

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GPA--TTS-yellow?style=for-the-badge)](https://huggingface.co/AutoArk-AI/GPA/tree/main/GPA_TTS/GPA_TTS_INT8) [![GPA](https://img.shields.io/badge/GPA-Main%20Repo-blue?style=for-the-badge)](https://github.com/AutoArk/GPA)

</div>

> **TL;DR** GPA-TTS is a standalone TTS runtime extracted from GPA, featuring zero-shot voice cloning, one of the **smallest memory footprints**, and ready-to-run scripts.

---

## 💡 Why GPA-TTS?

Based on usage patterns from our [HuggingFace Space demo](https://huggingface.co/spaces/AutoArk-AI/GPA_DEMO), **Text-to-Speech has been the most popular feature** among GPA users.

While our upcoming **GPA-v1.5** will be a larger and more capable unified model, we recognized the community's need for a **ultra-efficient, deployment-ready TTS solution**. 

Thus, we distilled the TTS component from GPA into this standalone package.

**GPA-TTS** is designed with edge deployment in mind:

- **Compact**: INT8 quantization + ONNX runtime enables **one of the smallest memory footprints among open-source TTS solutions**
- **Voice Cloning**: Zero-shot cloning from a short reference audio
- **Self-Contained**: No external LLM server required—everything runs locally
- **Production-Ready**: REST API with voice registration and management

We hope this release serves developers building voice applications on resource-constrained devices.

---

## Table of Contents

<div align="center">

| [📥 Download](#-download) | [🧹 Environment Setup](#-environment-setup) | [🔊 CLI Synthesis](#-cli-synthesis) | [🌐 Local Service](#-local-service) | [📡 API Reference](#-api-reference) |
| :---: | :---: | :---: | :---: | :---: |

</div>

---

## 📥 Download

Download the model files from Hugging Face:

**[🤗 AutoArk-AI/GPA/GPA_TTS/GPA_TTS_INT8](https://huggingface.co/AutoArk-AI/GPA/tree/main/GPA_TTS/GPA_TTS_INT8)**

After downloading, your local directory should look like this:

```text
GPA_TTS/
├── model/
│   ├── qwen_int4_ort/
│   ├── reference/
│   │   └── 038142_global_tokens.npy
│   ├── runtime_manifest.json
│   ├── spark_detokenizer_int8.onnx
│   └── spark_detokenizer_int8.onnx.data
├── voice/
│   └── spark_tokenizer_model/
│       ├── config.json
│       ├── config.yaml
│       ├── model.safetensors
│       └── wav2vec2-large-xlsr-53/
├── voices/
│   ├── items/
│   │   └── default/
│   │       ├── global_tokens.npy
│   │       └── meta.json
│   └── registry.json
├── requirements.txt
├── service.py
├── spark_tokenizer_runtime/
├── static/
├── tts_realtime_int8_light.py
├── voice_registration.py
└── voice_registry.py
```

> **⚠️ Important**: All three directories (`model/`, `voice/`, `voices/`) must be present. If any are missing, the runtime will not start correctly.

---

## 🧹 Environment Setup

Create a Python environment and install dependencies:

```bash
python3.12 -m venv .venv_tts_onnx
source .venv_tts_onnx/bin/activate
pip install -r requirements.txt
```

---

## 🔊 CLI Synthesis

Run a quick smoke test with the bundled default voice:

```bash
python tts_realtime_int8_light.py \
  --text "Hello, this is a test." \
  --global-token-path voices/items/default/global_tokens.npy \
  --output-path ./hello.wav
```

**Expected output:**
- `hello.wav` generated in the current directory
- Memory checkpoints and synthesis summary printed to console

> **💡 Tip**: If you encounter missing-file errors, verify that the HF assets were restored to the exact paths shown in the directory structure above.

---

## 🌐 Local Service

Start the FastAPI-based local server:

```bash
python -m uvicorn service:app --host 127.0.0.1 --port 8021
```

Once running, access:

| Page | URL |
| :--- | :--- |
| **Home** | http://127.0.0.1:8021/ |
| **Chinese Demo** | http://127.0.0.1:8021/static/demo_page_cn.html |
| **English Demo** | http://127.0.0.1:8021/static/demo_page_en.html |

---

## 📡 API Reference

### Health Check

```bash
curl http://127.0.0.1:8021/api/health
```

### Memory Usage

Returns current and peak memory usage of the service process—useful for monitoring resource consumption on edge devices.

```bash
curl http://127.0.0.1:8021/api/memory
```

### List Registered Voices

```bash
curl http://127.0.0.1:8021/api/voices
```

### Text-to-Speech

```bash
curl -X POST http://127.0.0.1:8021/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "voice_name": "default",
    "temperature": 0.3,
    "repetition_penalty": 1.2,
    "max_new_tokens": 512,
    "do_sample": false
  }'
```

### Register Voice (Local Path)

```bash
curl -X POST http://127.0.0.1:8021/api/voices/register-path \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-voice",
    "audio_path": "./reference_audio/sample.wav",
    "overwrite": false
  }'
```

### Register Voice (File Upload)

```bash
curl -X POST http://127.0.0.1:8021/api/voices/register-upload \
  -F "name=my-voice" \
  -F "overwrite=false" \
  -F "audio=@./reference_audio/sample.wav"
```

---

## 🔗 Related

- **[GPA Main Repository](https://github.com/AutoArk/GPA)** — Unified model for ASR, TTS, and Voice Conversion
- **[GPA Paper (ArXiv)](https://arxiv.org/abs/2601.10770)** — Technical details and evaluation
- **[Interactive Demo](https://huggingface.co/spaces/AutoArk-AI/GPA_DEMO)** — Try GPA online