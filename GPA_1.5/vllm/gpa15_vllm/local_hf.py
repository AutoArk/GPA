from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from transformers import AutoTokenizer, WhisperFeatureExtractor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@lru_cache(maxsize=16)
def load_local_module(model_dir: str, filename: str, module_prefix: str) -> ModuleType:
    module_path = Path(model_dir).expanduser().resolve() / filename
    if not module_path.is_file():
        raise FileNotFoundError(f"Cannot find local GPA module: {module_path}")

    module_name = f"{module_prefix}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import local GPA module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=8)
def load_local_processor(model_dir: str) -> Any:
    model_path = Path(model_dir).expanduser().resolve()
    module = load_local_module(
        str(model_path),
        "processing_arkasr.py",
        "gpa15_local_processing",
    )
    processor_cls = module.ArkasrProcessor

    processor_config_path = model_path / "processor_config.json"
    processor_config: dict[str, Any] = {}
    if processor_config_path.is_file():
        with processor_config_path.open("r", encoding="utf-8") as f:
            processor_config = json.load(f)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    for key, value in (processor_config.get("feature_extractor_config") or {}).items():
        if hasattr(feature_extractor, key):
            setattr(feature_extractor, key, value)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    for key, value in (processor_config.get("tokenizer_config") or {}).items():
        if hasattr(tokenizer, key):
            setattr(tokenizer, key, value)

    return processor_cls(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        merge_factor=int(processor_config.get("merge_factor", 4)),
        audio_token=str(processor_config.get("audio_token", "<|audio|>")),
        audio_dtype=str(processor_config.get("audio_dtype", "float32")),
    )
