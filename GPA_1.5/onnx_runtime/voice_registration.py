import gc
from pathlib import Path

import numpy as np
import psutil
import torch

from spark_tokenizer_runtime.spark_tokenizer import SparkTokenizer


class MemoryMonitor:
    def __init__(self, sample_interval_s: float = 0.05):
        self.process = psutil.Process()
        self.sample_interval_s = sample_interval_s
        self.peak_rss = self.process.memory_info().rss
        self._running = False
        self._thread = None

    def current_rss(self) -> int:
        return self.process.memory_info().rss

    def _run(self) -> None:
        import threading

        event = getattr(self, "_stop_event", None)
        while event is not None and not event.wait(self.sample_interval_s):
            self.peak_rss = max(self.peak_rss, self.current_rss())

    def start(self) -> None:
        import threading

        self._stop_event = threading.Event()
        self.peak_rss = self.current_rss()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        event = getattr(self, "_stop_event", None)
        if event is not None:
            event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss = max(self.peak_rss, self.current_rss())


def extract_global_tokens(*, tokenizer_model_dir: Path, audio_path: Path, device: str = "cpu") -> np.ndarray:
    tokenizer = SparkTokenizer(model_path=str(tokenizer_model_dir), device=device)
    try:
        result = tokenizer.tokenize([str(audio_path)])
        return np.asarray(result["global_tokens"].detach().cpu().long().numpy(), dtype=np.int64)
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
