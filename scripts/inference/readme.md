# GPA Inference Script / GPA 推理脚本

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

This script (`gpa_inference.py`) allows you to run inference with the GPA model for various audio tasks, including Speech-to-Text (STT), Text-to-Speech (TTS), and Voice Conversion (VC).

### Usage

Using `uv` (Recommended):
```bash
uv run gpa_inference.py --task <task_name> [options]
```

Or using standard python:
```bash
python gpa_inference.py --task <task_name> [options]
```

### Tasks

The script supports the following tasks:

*   `stt`: Speech-to-Text
*   `tts-a`: Text-to-Speech (Voice Cloning)
*   `vc`: Voice Conversion

### Examples

#### 1. Speech-to-Text (STT)
Transcribe an audio file.

```bash
# Using uv
uv run gpa_inference.py --task stt --src_audio_path "test_audio/000.wav"

# Or python
python gpa_inference.py --task stt --src_audio_path "test_audio/000.wav"
```

#### 2. Text-to-Speech (TTS)

Generate speech from text using a reference audio for timbre.

```bash
# Using uv
uv run gpa_inference.py --task tts-a --text "Hello world" --ref_audio_path "test_audio/000.wav"

# Or python
python gpa_inference.py --task tts-a --text "Hello world" --ref_audio_path "test_audio/000.wav"
```

#### 3. Voice Conversion (VC)
Convert the voice of a source audio to match the timbre of a reference audio.

```bash
# Using uv
uv run gpa_inference.py --task vc \
    --src_audio_path "test_audio/vc_src.wav" \
    --ref_audio_path "test_audio/000.wav"

# Or python
python gpa_inference.py --task vc \
    --src_audio_path "test_audio/vc_src.wav" \
    --ref_audio_path "test_audio/000.wav"
```

### Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--task` | **Required.** Task to run (`stt`, `tts-a`, `vc`). | - |
| `--device` | Device to use (e.g., `cuda:0`, `cpu`). | `cuda:3` (if available) |
| `--output_dir` | Directory to save output files. | `.` |
| `--ref_audio_path` | Path to the reference audio file (used for timbre cloning). | `test_audio/000.wav` |
| `--src_audio_path` | Path to the source audio file (used for VC). | `test_audio/vc_src.wav` |
| `--text` | Text content for TTS tasks. | (Demo defaults) |
| `--tokenizer_path` | Path to GLM4 tokenizer. | `/nasdata/model/gpa/glm-4-voice-tokenizer` |
| `--gpa_model_path` | Path to GPA model. | `/nasdata/model/gpa` |

---

<a name="chinese"></a>
## 中文

该脚本 (`gpa_inference.py`) 用于使用 GPA 模型执行各种音频推理任务，包括语音转文字 (STT)、文字转语音 (TTS) 和语音转换 (VC)。

### 用法

使用 `uv` (推荐):
```bash
uv run gpa_inference.py --task <任务名称> [选项]
```

或者使用标准 python:
```bash
python gpa_inference.py --task <任务名称> [选项]
```

### 任务列表

脚本支持以下任务：

*   `stt`: 语音转文字
*   `tts-a`: 文字转语音
*   `vc`: 语音转换

### 示例

#### 1. 语音转文字 (STT)
对音频文件进行转写。

```bash
# 使用 uv
uv run gpa_inference.py --task stt --ref_audio_path "test_audio/000.wav"

# 或者 python
python gpa_inference.py --task stt --ref_audio_path "test_audio/000.wav"
```

#### 2. 文字转语音 (TTS)

使用参考音频的音色将文本转换为语音。

```bash
# 使用 uv
uv run gpa_inference.py --task tts-a --text "你好，世界" --ref_audio_path "test_audio/000.wav"

# 或者 python
python gpa_inference.py --task tts-a --text "你好，世界" --ref_audio_path "test_audio/000.wav"
```

#### 3. 语音转换 (VC)
将源音频的声音转换为匹配参考音频的音色。

```bash
# 使用 uv
uv run gpa_inference.py --task vc \
    --src_audio_path "test_audio/vc_src.wav" \
    --ref_audio_path "test_audio/000.wav"

# 或者 python
python gpa_inference.py --task vc \
    --src_audio_path "test_audio/vc_src.wav" \
    --ref_audio_path "test_audio/000.wav"
```

### 参数说明

| 参数 | 描述 | 默认值 |
| :--- | :--- | :--- |
| `--task` | **必填。** 要运行的任务 (`stt`, `tts-a`, `vc`)。 | - |
| `--device` | 使用的设备 (例如 `cuda:0`, `cpu`)。 | `cuda:3` (如果可用) |
| `--output_dir` | 保存输出文件的目录。 | `.` |
| `--ref_audio_path` | 参考音频文件的路径 (用于音色克隆)。 | `test_audio/000.wav` |
| `--src_audio_path` | 源音频文件的路径 (用于 VC)。 | `test_audio/vc_src.wav` |
| `--text` | TTS 任务的文本内容。 | (演示默认值) |
| `--tokenizer_path` | GLM4 tokenizer 的路径。 | `/nasdata/model/gpa/glm-4-voice-tokenizer` |
| `--gpa_model_path` | GPA 模型的路径。 | `/nasdata/model/gpa` |