from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import transformers

from .assets import DEFAULT_HF_CACHE_DIR


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    audio_tokenizer_path: Optional[str] = field(default=None)
    attn_impl: str = field(
        default="auto",
        metadata={"help": "Attention backend: auto, flash_attention_2, sdpa, or eager."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training JSONL file."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the evaluation JSONL file."})
    hf_cache_dir: str = field(
        default=str(DEFAULT_HF_CACHE_DIR),
        metadata={"help": "Hugging Face datasets cache directory."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=8192)
    remove_unused_columns: bool = field(default=False)
    use_lora: bool = field(default=False)
    max_audio_seconds: int = field(default=30)
    sampling_rate: int = field(default=16000)
    tts_max_semantic_tokens: int = field(default=1024)
    tts_missing_token_policy: str = field(default="error")


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False