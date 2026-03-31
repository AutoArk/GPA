import argparse
import gc
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og
import psutil
import soundfile as sf


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_GLOBAL_TOKEN_PATH = "reference/038142_global_tokens.npy"
EOS_TOKEN_IDS = {151645}


class MemoryMonitor:
    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss = 0

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            time.sleep(self.interval_s)

    def start(self) -> None:
        self.peak_rss = self.process.memory_info().rss
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def current_rss(self) -> int:
        return self.process.memory_info().rss


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def log_memory(label: str, monitor: MemoryMonitor) -> None:
    print(f"[memory] {label}: rss={format_bytes(monitor.current_rss())}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the packaged GPA TTS INT8 runtime with resident models and selectable voices."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=script_dir / "model",
        help="Packaged model directory.",
    )
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument(
        "--global-token-path",
        type=Path,
        default=None,
        help="Optional override for the reference global token .npy file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=script_dir / "output_tts_int8_light.wav",
        help="Output wav path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        help="Enable sampling.",
    )
    parser.add_argument(
        "--no-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling.",
    )
    parser.set_defaults(do_sample=False)
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=None,
        help="Override ONNX Runtime intra-op thread count.",
    )
    return parser.parse_args()


def load_manifest(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / "runtime_manifest.json").read_text())


def normalize_global_tokens(global_tokens: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    global_tokens = np.asarray(global_tokens, dtype=np.int64)
    if global_tokens.ndim == 1:
        global_tokens = global_tokens[np.newaxis, np.newaxis, :]
    elif global_tokens.ndim == 2:
        global_tokens = global_tokens[:, np.newaxis, :]
    if tuple(global_tokens.shape) != expected_shape:
        raise ValueError(f"Unexpected global token shape: {tuple(global_tokens.shape)} != {expected_shape}")
    return np.ascontiguousarray(global_tokens)


def load_semantic_token_id_map(bundle_dir: Path) -> dict[int, int]:
    added_tokens_path = bundle_dir / "qwen_int4_ort" / "added_tokens.json"
    payload = json.loads(added_tokens_path.read_text())
    semantic_id_map: dict[int, int] = {}
    for token_text, token_id in payload.items():
        match = re.fullmatch(r"<\|bicodec_semantic_(\d+)\|>", token_text)
        if match:
            semantic_id_map[int(token_id)] = int(match.group(1))
    if not semantic_id_map:
        raise RuntimeError(f"No bicodec semantic tokens found in: {added_tokens_path}")
    return semantic_id_map


def build_prompt(tokenizer, text: str, global_tokens: np.ndarray, global_token_offset: int) -> str:
    shifted_global_ids = np.asarray(global_tokens.reshape(-1) + global_token_offset, dtype=np.int32)
    global_text = tokenizer.decode(shifted_global_ids)
    return (
        f"<|start_global_token|>{global_text}<|end_global_token|>"
        f"<|start_content|>{text}<|end_content|>"
    )


def create_detokenizer_session(model_path: Path, intra_op_threads: Optional[int]) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    if intra_op_threads is not None:
        session_options.intra_op_num_threads = intra_op_threads
        session_options.inter_op_num_threads = max(1, intra_op_threads // 2)
    return ort.InferenceSession(str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"])


def create_qwen_generator(model, input_ids: np.ndarray, args: argparse.Namespace):
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=int(input_ids.shape[1] + args.max_new_tokens),
        do_sample=args.do_sample,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    generator = og.Generator(model, params)
    generator.append_tokens(np.asarray(input_ids, dtype=np.int32).reshape(-1))
    return generator


def run_generation(generator) -> tuple[np.ndarray, list[int]]:
    last_next_tokens: list[int] = []
    while not generator.is_done():
        generator.generate_next_token()
        last_next_tokens = np.asarray(generator.get_next_tokens(), dtype=np.int64).tolist()
    return np.asarray(generator.get_sequence(0), dtype=np.int64), last_next_tokens


def one_pole_high_pass(wav: np.ndarray, sample_rate: int, cutoff_hz: float = 40.0) -> np.ndarray:
    if wav.size == 0:
        return wav
    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = rc / (rc + dt)
    y = np.empty_like(wav, dtype=np.float32)
    prev_y = 0.0
    prev_x = float(wav[0])
    for idx, x in enumerate(wav.astype(np.float32, copy=False)):
        current = alpha * (prev_y + float(x) - prev_x)
        y[idx] = current
        prev_y = current
        prev_x = float(x)
    return y


def soft_noise_gate(
    wav: np.ndarray,
    sample_rate: int,
    floor_percentile: float = 25.0,
    threshold_scale: float = 1.8,
    smooth_ms: float = 12.0,
) -> np.ndarray:
    if wav.size == 0:
        return wav
    amplitude = np.abs(wav).astype(np.float32, copy=False)
    window = max(1, int(sample_rate * smooth_ms / 1000.0))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    envelope = np.convolve(amplitude, kernel, mode="same")
    noise_floor = float(np.percentile(envelope, floor_percentile))
    threshold = max(noise_floor * threshold_scale, 1e-5)
    gain = np.clip((envelope - noise_floor) / max(threshold - noise_floor, 1e-5), 0.0, 1.0)
    gain = 0.15 + 0.85 * gain
    return wav * gain.astype(np.float32, copy=False)


def spectral_gate_denoise(
    wav: np.ndarray,
    sample_rate: int,
    frame_ms: float = 32.0,
    hop_ms: float = 8.0,
    noise_percentile: float = 18.0,
    threshold_scale: float = 1.35,
    floor_scale: float = 0.35,
) -> np.ndarray:
    if wav.size == 0:
        return wav
    frame_length = max(256, int(sample_rate * frame_ms / 1000.0))
    hop_length = max(64, int(sample_rate * hop_ms / 1000.0))
    window = np.hanning(frame_length).astype(np.float32)
    if wav.size < frame_length:
        padded = np.pad(wav, (0, frame_length - wav.size))
    else:
        pad = (hop_length - (wav.size - frame_length) % hop_length) % hop_length
        padded = np.pad(wav, (0, pad))
    num_frames = 1 + (padded.size - frame_length) // hop_length
    spec = np.empty((frame_length // 2 + 1, num_frames), dtype=np.complex64)
    for idx in range(num_frames):
        start = idx * hop_length
        frame = padded[start:start + frame_length] * window
        spec[:, idx] = np.fft.rfft(frame).astype(np.complex64)
    magnitude = np.abs(spec)
    phase = np.angle(spec)
    noise_profile = np.percentile(magnitude, noise_percentile, axis=1, keepdims=True)
    threshold = np.maximum(noise_profile * threshold_scale, 1e-6)
    gain = np.clip((magnitude - noise_profile) / np.maximum(threshold - noise_profile, 1e-6), 0.0, 1.0)
    gain = floor_scale + (1.0 - floor_scale) * gain
    filtered = magnitude * gain * np.exp(1j * phase)
    reconstructed = np.zeros(padded.size, dtype=np.float32)
    window_sum = np.zeros(padded.size, dtype=np.float32)
    for idx in range(num_frames):
        start = idx * hop_length
        frame = np.fft.irfft(filtered[:, idx], n=frame_length).astype(np.float32)
        reconstructed[start:start + frame_length] += frame * window
        window_sum[start:start + frame_length] += window * window
    valid = window_sum > 1e-6
    reconstructed[valid] /= window_sum[valid]
    return reconstructed[: wav.size]


def postprocess_audio(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    if wav.size == 0:
        return wav
    wav = wav.astype(np.float32, copy=False)
    wav = wav - wav.mean()
    wav = one_pole_high_pass(wav, sample_rate=sample_rate, cutoff_hz=40.0)
    wav = soft_noise_gate(wav, sample_rate=sample_rate)
    wav = spectral_gate_denoise(wav, sample_rate=sample_rate)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.99:
        wav = wav / peak * 0.99
    return wav


class PackagedTTSRuntime:
    def __init__(self, bundle_dir: Path, default_global_token_path: Optional[Path] = None, intra_op_threads: Optional[int] = None):
        self.bundle_dir = bundle_dir.resolve()
        self.intra_op_threads = intra_op_threads
        self.manifest = load_manifest(self.bundle_dir)
        self.semantic_token_id_map = load_semantic_token_id_map(self.bundle_dir)
        self.default_global_token_path = (
            default_global_token_path.resolve()
            if default_global_token_path is not None
            else (self.bundle_dir / DEFAULT_GLOBAL_TOKEN_PATH).resolve()
        )
        self.qwen_model = og.Model(str(self.bundle_dir / "qwen_int4_ort"))
        self.tokenizer = og.Tokenizer(self.qwen_model)
        self.detok_session = create_detokenizer_session(self.bundle_dir / "spark_detokenizer_int8.onnx", intra_op_threads)

    def close(self) -> None:
        if hasattr(self, "detok_session"):
            del self.detok_session
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "qwen_model"):
            del self.qwen_model
        gc.collect()

    def load_global_tokens(self, global_token_path: Optional[Path] = None) -> np.ndarray:
        token_path = global_token_path.resolve() if global_token_path is not None else self.default_global_token_path
        return normalize_global_tokens(np.load(token_path), tuple(self.manifest["global_tokens_shape"]))

    def synthesize_to_file(
        self,
        *,
        text: str,
        output_path: Path,
        global_token_path: Optional[Path] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        do_sample: bool = False,
    ) -> dict:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        monitor = MemoryMonitor()
        monitor.start()
        memory_checkpoints: dict[str, int] = {}

        def capture_memory(label: str) -> None:
            memory_checkpoints[label] = monitor.current_rss()
            log_memory(label, monitor)

        generator = None
        generated_ids = None
        completion_ids = None
        prompt_ids = None
        try:
            capture_memory("request_start")
            selected_global_token_path = global_token_path.resolve() if global_token_path is not None else self.default_global_token_path
            global_tokens = self.load_global_tokens(global_token_path)
            capture_memory("after_global_token_load")

            prompt = build_prompt(
                tokenizer=self.tokenizer,
                text=text,
                global_tokens=global_tokens,
                global_token_offset=self.manifest["global_token_offset"],
            )
            prompt_ids = np.asarray(self.tokenizer.encode(prompt), dtype=np.int32).reshape(1, -1)
            capture_memory("after_prompt_build")

            runtime_args = argparse.Namespace(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )
            generator = create_qwen_generator(self.qwen_model, prompt_ids, runtime_args)
            generated_ids, last_next_tokens = run_generation(generator)
            capture_memory("after_generation")

            prompt_length = int(prompt_ids.shape[1])
            completion_ids = generated_ids[prompt_length:]
            last_token_id = int(completion_ids[-1]) if completion_ids.size else None
            last_emitted_token_id = int(last_next_tokens[-1]) if last_next_tokens else last_token_id
            stopped_by_eos = last_emitted_token_id in EOS_TOKEN_IDS if last_emitted_token_id is not None else False
            hit_max_new_tokens = int(completion_ids.shape[0]) >= int(max_new_tokens)
            audio_ids = [
                self.semantic_token_id_map[int(token_id)]
                for token_id in completion_ids.tolist()
                if int(token_id) in self.semantic_token_id_map
            ]
            if not audio_ids:
                raise RuntimeError("No generated bicodec semantic tokens were found in the model output.")

            del generated_ids
            del completion_ids
            del prompt_ids
            del generator
            generated_ids = None
            completion_ids = None
            prompt_ids = None
            generator = None
            gc.collect()
            capture_memory("after_kv_cache_release")

            semantic_tokens = np.asarray(audio_ids, dtype=np.int64)[np.newaxis, :]
            audio = self.detok_session.run(
                None,
                {"semantic_tokens": semantic_tokens, "global_tokens": global_tokens},
            )[0]
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size > 0:
                audio = audio[: len(audio_ids) * self.manifest["latent_hop_length"]]
                audio = postprocess_audio(audio, sample_rate=int(self.manifest["sample_rate"]))
            sf.write(output_path, audio, int(self.manifest["sample_rate"]))
            capture_memory("after_detokenize")

            print("Used SparkDetokenizer ONNX artifact: spark_detokenizer_int8.onnx")
            print("Denoise: enabled mode=moderate")
            print(
                "Generation stop:"
                f" stopped_by_eos={stopped_by_eos}"
                f" hit_max_new_tokens={hit_max_new_tokens}"
                f" last_token_id={last_token_id if last_token_id is not None else 'none'}"
                f" last_emitted_token_id={last_emitted_token_id if last_emitted_token_id is not None else 'none'}"
            )
            print(f"Generated {len(audio_ids)} semantic tokens")
            print(f"Saved wav to: {output_path}")
            return {
                "output_path": str(output_path),
                "sample_rate": int(self.manifest["sample_rate"]),
                "num_semantic_tokens": len(audio_ids),
                "stopped_by_eos": stopped_by_eos,
                "hit_max_new_tokens": hit_max_new_tokens,
                "last_token_id": last_token_id,
                "last_emitted_token_id": last_emitted_token_id,
                "selected_global_token_path": str(selected_global_token_path),
                "memory_checkpoints": memory_checkpoints,
                "peak_rss_bytes": monitor.peak_rss,
                "models_resident": True,
                "kv_cache_released": True,
            }
        finally:
            if generator is not None:
                del generator
            if generated_ids is not None:
                del generated_ids
            if completion_ids is not None:
                del completion_ids
            if prompt_ids is not None:
                del prompt_ids
            gc.collect()
            monitor.stop()
            print(f"[memory] peak_rss={format_bytes(monitor.peak_rss)}")


def synthesize_to_file(
    *,
    bundle_dir: Path,
    text: str,
    output_path: Path,
    global_token_path: Optional[Path] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    repetition_penalty: float = 1.2,
    do_sample: bool = False,
    intra_op_threads: Optional[int] = None,
) -> dict:
    runtime = PackagedTTSRuntime(
        bundle_dir=bundle_dir,
        default_global_token_path=global_token_path,
        intra_op_threads=intra_op_threads,
    )
    try:
        return runtime.synthesize_to_file(
            text=text,
            output_path=output_path,
            global_token_path=global_token_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
    finally:
        runtime.close()


def synthesize(args: argparse.Namespace) -> dict:
    return synthesize_to_file(
        bundle_dir=args.bundle_dir,
        text=args.text,
        output_path=args.output_path,
        global_token_path=args.global_token_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        intra_op_threads=args.intra_op_threads,
    )


def main() -> None:
    synthesize(parse_args())


if __name__ == "__main__":
    main()
