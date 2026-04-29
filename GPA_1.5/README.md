# GPA v1.5

> **TL;DR** This directory is the code home for GPA v1.5 training, native inference, and ONNX runtime workflows. Start with [docs/train.md](docs/train.md) for fine-tuning, [docs/infer.md](docs/infer.md) for native PyTorch inference, or [onnx_runtime/README.md](onnx_runtime/README.md) for ONNX CLI/service deployment.

GPA v1.5 contains 3 tracks:

- **Native train** for fine-tuning and continued training GPA-v1.5 with Hugging Face `Trainer`.
- **Native infer** for direct Hugging Face and PyTorch execution of GPA-v1.5's inference.
- **ONNX runtime** for local CLI inference, FastAPI service deployment, browser UI testing, and runtime-focused validation.

## 📥 Download Index

Large model assets are hosted separately from this code tree. Download the GPA v1.5 checkpoint first, then place it in the recommended local layout below.

<div align="center">

| Asset | Recommended Local Path | Download |
| :--- | :--- | :---: |
| **🤗 GPA-v1.5 Hugging Face checkpoint** | `GPA-v1.5-HF/GPA-v1.5` | **[Download →](https://huggingface.co/AutoArk-AI/GPA-v1.5)** |
| **🎙️ Spark tokenizer assets** | `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model` | Included in the checkpoint |
| **⚙️ GPA-v1.5 ONNX runtime assets** | `GPA-v1.5-HF/GPA-v1.5-onnx-runtime` | **[Download →](https://huggingface.co/AutoArk-AI/GPA-v1.5-onnx-runtime)** |

</div>

> **💡 Recommended placement**: Create one local asset folder named `GPA-v1.5-HF/`, then put the downloaded Hugging Face repos inside it exactly like this:
>
> ```text
> GPA-v1.5-HF/
> ├── GPA-v1.5/
> │   └── spark_tokenizer_model/
> └── GPA-v1.5-onnx-runtime/
> ```
>
> With this layout, native train/infer can find `GPA-v1.5/` and its `spark_tokenizer_model/`, and ONNX runtime can find `GPA-v1.5-onnx-runtime/` without extra environment variables.

## 🚀 Where To Start

Choose the path that matches your goal:

- **I want to fine-tune or continue training GPA v1.5:**
[docs/train.md](docs/train.md)
- **I want the direct model behavior baseline:**
[docs/infer.md](docs/infer.md)
- **I want ONNX CLI inference, FastAPI service, or the browser UI:**
[onnx_runtime/README.md](onnx_runtime/README.md)

## 🧭 Recommended Local Layout

This repo does not bundle the large model assets directly. For the least configuration, keep the downloaded checkpoint repos side by side:

```text
GPA-v1.5/
GPA-v1.5-HF/
  GPA-v1.5/
    spark_tokenizer_model/
  GPA-v1.5-onnx-runtime/
```

What each path is used for:

- `GPA-v1.5-HF/GPA-v1.5`: native PyTorch train/infer Hugging Face checkpoint
- `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model`: Spark tokenizer assets used by native TTS
- `GPA-v1.5-HF/GPA-v1.5-onnx-runtime`: ONNX CLI/service/browser UI asset bundle

That means the smoke tests in [docs/infer.md](docs/infer.md), [docs/train.md](docs/train.md), and [onnx_runtime/README.md](onnx_runtime/README.md) can run without editing source paths.

If your assets live elsewhere, use CLI flags or environment variables instead of modifying the source tree.


## 🔍 Quick Tour

- `train.py`
  Thin training CLI entry that resolves local defaults and forwards into the internal training package.
- `train.sh`
  Thin shell wrapper for local training runs. It can launch plain Python by default and DeepSpeed when requested.
- `training/`
  Internal training modules for dataset loading, collators, save helpers, and the main training runner.
- `requirements.train.txt`
  Training dependency list.
- `infer.py`
  Native CLI entry for ASR and TTS. (VC planned for future release.)
- `infer.sh`
  Thin shell wrapper that injects the expected local asset paths.
- `inference/`
  Internal modules for model loading, prompt construction, decode policies, and task-specific execution.
- `tts_han_char_tokenizer.py`
  Shared TTS text normalization logic used to keep mixed Han and Latin text behavior stable.
- `configs/`
  Example DeepSpeed config and hostfile template for multi-node launches.
- `docs/`
  User-facing guides for this directory.
- `spark_tokenizer_runtime/`
  Spark tokenizer and detokenizer support used by native TTS paths.
- `onnx_runtime/`
  ONNX inference scripts, FastAPI service, browser UI, voice registration, export/quantization scripts, and runtime validation tools.

## 📖 Documentation Map

- **Training guide**: [docs/train.md](docs/train.md)
- **Native infer guide**: [docs/infer.md](docs/infer.md)
- **ONNX runtime guide**: [onnx_runtime/README.md](onnx_runtime/README.md)
