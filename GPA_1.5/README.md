# GPA v1.5

> **TL;DR** This directory is the code home for GPA v1.5 training and native inference workflows. Start with [docs/train.md](docs/train.md) for fine-tuning or [docs/infer.md](docs/infer.md) for native PyTorch inference.

GPA v1.5 contains 3 tracks:

- **Native train** for fine-tuning and continued training GPA-v1.5 with Hugging Face `Trainer`.
- **Native infer** for direct Hugging Face and PyTorch execution of GPA-v1.5's inference.
- **ONNX runtime(Coming Soon)** enable seamless inference for ONNX-formatted models — available as a hosted service or local CLI toolkit.

## 📥 Download Index

Large model assets are hosted separately from this code tree. Download the GPA v1.5 checkpoint first, then place it in the recommended local layout below.

<div align="center">

| Asset | Recommended Local Path | Download |
| :--- | :--- | :---: |
| **🤗 GPA-v1.5 Hugging Face checkpoint** | `GPA-v1.5-HF/GPA-v1.5` | **[Download →](https://huggingface.co/AutoArk-AI/GPA-v1.5)** |
| **🎙️ Spark tokenizer assets** | `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model` | Included in the checkpoint |

</div>

> **💡 Tip**: Keep the downloaded files together. Native train and native infer both auto-discover the model and Spark tokenizer when this layout is preserved.

## 🚀 Where To Start

Choose the path that matches your goal:

- **I want to fine-tune or continue training GPA v1.5:**
[docs/train.md](docs/train.md)
- **I want the direct model behavior baseline:**
[docs/infer.md](docs/infer.md)

## 🧭 Recommended Local Layout

This repo does not bundle the large model assets directly. The expected local sibling layout is:

```text
GPA-v1.5/
GPA-v1.5-HF/
  GPA-v1.5/
    spark_tokenizer_model/
```

With that layout in place:

- native infer automatically discovers `GPA-v1.5-HF/GPA-v1.5`
- native infer automatically discovers `GPA-v1.5-HF/GPA-v1.5/spark_tokenizer_model`

That means most local smoke tests can run without editing code or exporting extra variables.

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

## 📖 Documentation Map

- **Training guide**: [docs/train.md](docs/train.md)
- **Native infer guide**: [docs/infer.md](docs/infer.md)
