from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[idx]


async def post_once(
    client: httpx.AsyncClient,
    url: str,
    audio_path: Path,
    timeout: float,
    mode: str,
    text: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    with audio_path.open("rb") as f:
        if mode == "tts":
            files = {"ref_file": (audio_path.name, f, "application/octet-stream")}
            data = {"text": text}
        else:
            files = {"file": (audio_path.name, f, "application/octet-stream")}
            data = None
        resp = await client.post(url, data=data, files=files, timeout=timeout)
    elapsed = time.perf_counter() - started
    resp.raise_for_status()
    if mode == "tts":
        return {
            "latency_s": elapsed,
            "server_latency_s": resp.headers.get("x-gpa-latency-s"),
            "prompt_tokens": resp.headers.get("x-gpa-prompt-tokens"),
            "semantic_tokens": resp.headers.get("x-gpa-semantic-tokens"),
            "audio_bytes": len(resp.content),
            "text": "",
        }
    data = resp.json()
    return {
        "latency_s": elapsed,
        "server_latency_s": data.get("latency_s"),
        "prompt_tokens": data.get("prompt_tokens"),
        "text": data.get("text", ""),
    }


async def run(args: argparse.Namespace) -> None:
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(args.warmup):
            await post_once(client, args.url, audio_path, args.timeout, args.mode, args.text)

        sem = asyncio.Semaphore(args.concurrency)
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        async def worker(_: int) -> None:
            async with sem:
                try:
                    results.append(await post_once(client, args.url, audio_path, args.timeout, args.mode, args.text))
                except Exception as exc:
                    errors.append(repr(exc))

        started = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(args.requests)))
        total = time.perf_counter() - started

    latencies = [float(item["latency_s"]) for item in results]
    server_latencies = [
        float(item["server_latency_s"])
        for item in results
        if item.get("server_latency_s") is not None
    ]

    print(f"requests={args.requests} concurrency={args.concurrency} ok={len(results)} errors={len(errors)}")
    print(f"wall_s={total:.3f} throughput_rps={len(results) / total if total else 0:.3f}")
    if latencies:
        print(
            "client_latency_s "
            f"mean={statistics.mean(latencies):.3f} "
            f"p50={percentile(latencies, 50):.3f} "
            f"p90={percentile(latencies, 90):.3f} "
            f"p95={percentile(latencies, 95):.3f} "
            f"p99={percentile(latencies, 99):.3f} "
            f"max={max(latencies):.3f}"
        )
    if server_latencies:
        print(
            "server_latency_s "
            f"mean={statistics.mean(server_latencies):.3f} "
            f"p50={percentile(server_latencies, 50):.3f} "
            f"p90={percentile(server_latencies, 90):.3f} "
            f"p95={percentile(server_latencies, 95):.3f} "
            f"p99={percentile(server_latencies, 99):.3f} "
            f"max={max(server_latencies):.3f}"
        )
    if results:
        print(f"sample_text={results[0].get('text', '')[:200]}")
        if args.mode == "tts":
            print(
                "sample_tts "
                f"audio_bytes={results[0].get('audio_bytes')} "
                f"semantic_tokens={results[0].get('semantic_tokens')}"
            )
    if errors:
        print("first_error=" + errors[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPA 1.5 vLLM ASR latency/concurrency")
    parser.add_argument("--url", default="http://127.0.0.1:18080/asr")
    parser.add_argument("--mode", choices=["asr", "tts"], default="asr")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", default="你好，世界。")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
