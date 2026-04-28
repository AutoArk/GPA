from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from spark_tokenizer_runtime.spark_detokenizer import SparkDeTokenizer
from spark_tokenizer_runtime.spark_tokenizer import SparkTokenizer
from tts_han_char_tokenizer import create_tts_han_char_processor


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_attn_impl(device: str, attn_impl: str) -> str:
    if attn_impl != "auto":
        return attn_impl
    if device.startswith("cuda"):
        return "flash_attention_2"
    if device == "mps":
        return "sdpa"
    return "eager"


def load_text_stack(model_path: Path, device: str, attn_impl: str):
    normalized_attn_impl = normalize_attn_impl(device=device, attn_impl=attn_impl)
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else None

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    tts_processor = create_tts_han_char_processor(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        attn_implementation=normalized_attn_impl,
        torch_dtype=torch_dtype,
    )
    model.eval().to(device)
    return tokenizer, processor, tts_processor, model, normalized_attn_impl


def load_spark_stack(audio_tokenizer_path: Path, device: str):
    spark_tokenizer = SparkTokenizer(str(audio_tokenizer_path), device=device)
    spark_detokenizer = SparkDeTokenizer(str(audio_tokenizer_path), device=device)
    return spark_tokenizer, spark_detokenizer
