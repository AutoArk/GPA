from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class Sample:
    id: str
    lang: str
    text: str
    prompt_audio: Path
    target_audio: Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def load_manifest(path: Path, limit: int) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample = Sample(
                id=str(row["id"]),
                lang=str(row["lang"]),
                text=str(row["text"]),
                prompt_audio=Path(row["prompt_audio"]),
                target_audio=Path(row["target_audio"]),
            )
            if sample.prompt_audio.is_file() and sample.target_audio.is_file():
                samples.append(sample)
            if len(samples) >= limit:
                break
    if not samples:
        raise RuntimeError(f"No usable samples in {path}")
    return samples


async def post_asr(
    client: httpx.AsyncClient,
    url: str,
    sample: Sample,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    with sample.target_audio.open("rb") as f:
        files = {"file": (sample.target_audio.name, f, "application/octet-stream")}
        resp = await client.post(url, files=files, timeout=timeout)
    elapsed = time.perf_counter() - started
    resp.raise_for_status()
    data = resp.json()
    return {
        "mode": "asr",
        "latency_s": elapsed,
        "server_latency_s": float(data.get("latency_s", 0.0)),
        "prompt_tokens": data.get("prompt_tokens"),
        "id": sample.id,
    }


async def post_tts_stream(
    client: httpx.AsyncClient,
    url: str,
    sample: Sample,
    timeout: float,
    max_new_tokens: int,
    chunk_semantic_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_byte_s: float | None = None
    total_bytes = 0
    with sample.prompt_audio.open("rb") as f:
        files = {"ref_file": (sample.prompt_audio.name, f, "application/octet-stream")}
        data = {
            "text": sample.text,
            "max_new_tokens": str(max_new_tokens),
            "chunk_semantic_tokens": str(chunk_semantic_tokens),
        }
        async with client.stream("POST", url, data=data, files=files, timeout=timeout) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                now = time.perf_counter()
                if first_byte_s is None:
                    first_byte_s = now - started
                total_bytes += len(chunk)
    elapsed = time.perf_counter() - started
    return {
        "mode": "tts",
        "latency_s": elapsed,
        "ttfb_s": first_byte_s if first_byte_s is not None else elapsed,
        "audio_bytes": total_bytes,
        "id": sample.id,
    }


def summarize(name: str, results: list[dict[str, Any]], errors: list[str], total_s: float) -> None:
    lat = [float(r["latency_s"]) for r in results]
    print(f"\n{name}: ok={len(results)} errors={len(errors)} wall_s={total_s:.3f} rps={len(results)/total_s if total_s else 0:.3f}")
    if lat:
        print(
            "latency_s "
            f"mean={statistics.mean(lat):.3f} "
            f"p50={percentile(lat, 50):.3f} "
            f"p90={percentile(lat, 90):.3f} "
            f"p95={percentile(lat, 95):.3f} "
            f"p99={percentile(lat, 99):.3f} "
            f"max={max(lat):.3f}"
        )
    ttfb = [float(r["ttfb_s"]) for r in results if "ttfb_s" in r]
    if ttfb:
        print(
            "ttfb_s "
            f"mean={statistics.mean(ttfb):.3f} "
            f"p50={percentile(ttfb, 50):.3f} "
            f"p90={percentile(ttfb, 90):.3f} "
            f"p95={percentile(ttfb, 95):.3f} "
            f"p99={percentile(ttfb, 99):.3f} "
            f"max={max(ttfb):.3f}"
        )
    if errors:
        print("first_error=" + errors[0])


async def run_phase(args: argparse.Namespace, samples: list[Sample], concurrency: int) -> tuple[list[dict[str, Any]], list[str], float]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    async with httpx.AsyncClient(trust_env=False) as client:
        async def worker(index: int, sample: Sample) -> None:
            mode = "asr" if args.mode == "asr" else "tts"
            if args.mode == "mixed":
                mode = "asr" if index % 2 == 0 else "tts"
            async with sem:
                try:
                    if mode == "asr":
                        result = await post_asr(client, args.asr_url, sample, args.timeout)
                    else:
                        result = await post_tts_stream(
                            client,
                            args.tts_stream_url,
                            sample,
                            args.timeout,
                            args.tts_max_new_tokens,
                            args.chunk_semantic_tokens,
                        )
                    results.append(result)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")

        started = time.perf_counter()
        await asyncio.gather(*(worker(i, sample) for i, sample in enumerate(samples)))
        total = time.perf_counter() - started

    return results, errors, total


async def run(args: argparse.Namespace) -> None:
    samples = load_manifest(Path(args.manifest).expanduser().resolve(), args.samples)
    for concurrency in args.concurrency:
        phase_samples = samples[: args.requests]
        if args.repeat:
            phase_samples = [samples[i % len(samples)] for i in range(args.requests)]
        print(f"\n=== mode={args.mode} concurrency={concurrency} requests={len(phase_samples)} ===")
        results, errors, total = await run_phase(args, phase_samples, concurrency)
        summarize(f"all@c{concurrency}", results, errors, total)
        asr_results = [r for r in results if r.get("mode") == "asr"]
        tts_results = [r for r in results if r.get("mode") == "tts"]
        if asr_results and tts_results:
            summarize(f"asr@c{concurrency}", asr_results, [], total)
            summarize(f"tts@c{concurrency}", tts_results, [], total)

        if args.stop_on_degradation:
            lat = [float(r.get("latency_s", 0.0)) for r in results]
            ttfb = [float(r.get("ttfb_s", 0.0)) for r in results if "ttfb_s" in r]
            if errors or (lat and percentile(lat, 95) >= args.max_p95_latency_s) or (
                ttfb and percentile(ttfb, 95) >= args.max_p95_ttfb_s
            ):
                print("stopping: degradation threshold reached")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pressure test GPA ASR/TTS with SeedTTS samples")
    parser.add_argument("--manifest", default="GPA_1.5/vllm/seedtts_256.jsonl")
    parser.add_argument("--mode", choices=["asr", "tts", "mixed"], default="mixed")
    parser.add_argument("--asr-url", default="http://127.0.0.1:18080/asr")
    parser.add_argument("--tts-stream-url", default="http://127.0.0.1:18080/tts/stream")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--tts-max-new-tokens", type=int, default=1024)
    parser.add_argument("--chunk-semantic-tokens", type=int, default=32)
    parser.add_argument("--stop-on-degradation", action="store_true")
    parser.add_argument("--max-p95-latency-s", type=float, default=30.0)
    parser.add_argument("--max-p95-ttfb-s", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
