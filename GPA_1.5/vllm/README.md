# GPA 1.5 vLLM ASR experiment

This directory contains a Docker Compose deployment and a small concurrency
benchmark for `/data2/model/AutoArk/GPA-v1_5-0_6B`.

The service is configured for local-only Hugging Face resources. The GPA
processor/audio modules are imported from the mounted model directory, and
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, plus `local_files_only=True` are
used for model-side tokenizer/config loading. If a required file is missing from
the mounted model directory, add it locally before starting the service.

## Start

```bash
cd /data3/yiming/GPA/GPA_1.5/vllm
docker compose up -d
curl -s http://127.0.0.1:18080/health
```

Useful environment overrides:

```bash
CUDA_VISIBLE_DEVICES=0 GPA15_VLLM_PORT=18080 VLLM_MAX_NUM_SEQS=16 docker compose up -d
```

## Request

```bash
curl -s -F "file=@/data3/yiming/GPA/scripts/inference/test_audio/000.wav" \
  http://127.0.0.1:18080/asr
```

## Benchmark

```bash
python GPA_1.5/vllm/bench_asr.py \
  --audio /data3/yiming/GPA/scripts/inference/test_audio/000.wav \
  --requests 32 \
  --concurrency 4
```

When benchmarking with the same short audio repeatedly, vLLM prefix/encoder
cache can make the measured latency much lower than an uncached mixed-audio
workload. Use a larger audio set for a production-shaped benchmark.
