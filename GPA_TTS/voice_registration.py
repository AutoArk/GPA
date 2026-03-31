import gc
from pathlib import Path

import numpy as np
import torch

from spark_tokenizer_runtime.spark_tokenizer import SparkTokenizer
from tts_realtime_int8_light import MemoryMonitor


class RegistrationResult(dict):
    pass


def extract_global_tokens(
    *,
    tokenizer_model_dir: Path,
    audio_path: Path,
    device: str = "cpu",
) -> np.ndarray:
    tokenizer = SparkTokenizer(model_path=str(tokenizer_model_dir), device=device)
    try:
        result = tokenizer.tokenize([str(audio_path)])
        global_tokens = result["global_tokens"].detach().cpu().long().numpy()
        return np.asarray(global_tokens, dtype=np.int64)
    finally:
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def register_voice_from_audio(
    *,
    tokenizer_model_dir: Path,
    registry,
    name: str,
    audio_path: Path,
    source_kind: str,
    source_label: str,
    overwrite: bool = False,
    device: str = "cpu",
) -> dict:
    monitor = MemoryMonitor()
    monitor.start()
    checkpoints: dict[str, int] = {}

    def capture(label: str) -> None:
        checkpoints[label] = monitor.current_rss()

    try:
        capture("start")
        global_tokens = extract_global_tokens(
            tokenizer_model_dir=tokenizer_model_dir,
            audio_path=audio_path,
            device=device,
        )
        capture("after_tokenize")
        voice = registry.register(
            name=name,
            global_tokens=global_tokens,
            source_kind=source_kind,
            source_label=source_label,
            overwrite=overwrite,
        )
        capture("after_registry_write")
        return {
            "voice": voice,
            "memory_checkpoints": checkpoints,
            "peak_rss_bytes": monitor.peak_rss,
        }
    finally:
        gc.collect()
        monitor.stop()
