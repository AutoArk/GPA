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

## TTS

The same vLLM engine can also generate BiCodec semantic tokens for TTS. The
service uses the mounted local BiCodec assets to extract reference global tokens
and detokenize the generated semantic tokens into a WAV response.

```bash
curl -s \
  -F "text=你好，世界" \
  -F "ref_file=@/data3/yiming/GPA/scripts/inference/test_audio/000.wav" \
  http://127.0.0.1:18080/tts \
  -o output_gpa_tts.wav
```

## Benchmark

```bash
python GPA_1.5/vllm/bench_asr.py \
  --audio /data3/yiming/GPA/scripts/inference/test_audio/000.wav \
  --requests 32 \
  --concurrency 4
```

TTS benchmark:

```bash
python GPA_1.5/vllm/bench_asr.py \
  --mode tts \
  --url http://127.0.0.1:18080/tts \
  --audio /data3/yiming/GPA/scripts/inference/test_audio/000.wav \
  --text "你好，世界。" \
  --requests 8 \
  --concurrency 2
```

When benchmarking with the same short audio repeatedly, vLLM prefix/encoder
cache can make the measured latency much lower than an uncached mixed-audio
workload. Use a larger audio set for a production-shaped benchmark.

## SeedTTS Pressure Test

Create a fixed 256-item manifest from `/data3/seedtts_testset`:

```bash
python GPA_1.5/vllm/make_seedtts_manifest.py \
  --root /data3/seedtts_testset \
  --output GPA_1.5/vllm/seedtts_256.jsonl \
  --num-samples 256
```

Run mixed ASR plus streaming TTS pressure tests:

```bash
python GPA_1.5/vllm/pressure_seedtts.py \
  --manifest GPA_1.5/vllm/seedtts_256.jsonl \
  --mode mixed \
  --samples 256 \
  --requests 256 \
  --concurrency 1 2 4 8 16 32 64 96 128 160 \
  --tts-max-new-tokens 768 \
  --chunk-semantic-tokens 32 \
  --stop-on-degradation \
  --max-p95-latency-s 30 \
  --max-p95-ttfb-s 10
```

The streaming TTS endpoint is `/tts/stream`; it returns raw PCM
`audio/L16; rate=16000; channels=1`. The pressure script measures TTS TTFB as
time to the first audio chunk and ASR latency as full request latency.

### Current Pressure Protocol

Environment:

- Service: one `gpa15-vllm` container serving both ASR and TTS from one vLLM
  `AsyncLLMEngine`.
- Model: local `/data2/model/AutoArk/GPA-v1_5-0_6B`.
- BiCodec: local `/data2/model/AutoArk/GPA/BiCodec`.
- Endpoint base URL: `http://127.0.0.1:18080`.
- External downloads are disabled by the compose environment and
  `local_files_only=True`.

Dataset:

- Source root: `/data3/seedtts_testset`.
- Manifest command: the `make_seedtts_manifest.py` command above.
- Random seed: `20260513`.
- Manifest size: 256 items.
- Observed language split for the generated manifest: 166 zh and 90 en.
- ASR uses each row's `target_audio`.
- TTS uses each row's `prompt_audio` as the voice reference and `text` as the
  synthesis target.

Load shape:

- Mode: `mixed`, alternating ASR and streaming TTS requests by request index.
- Requests per concurrency level: 256.
- Samples per run: 256, no repeat flag.
- Concurrency ladder: `1 2 4 8 16 32 64 96 128 160`.
- TTS generation cap: `--tts-max-new-tokens 768`.
- Streaming detokenization chunk: `--chunk-semantic-tokens 32`.
- Request timeout: 900 seconds.
- Stop rule: stop on any error, all-request p95 latency >= 30 seconds, or TTS
  p95 TTFB >= 10 seconds.

Metrics:

- ASR latency is measured client-side from request start to full JSON response.
- TTS total latency is measured client-side from request start to stream end.
- TTS TTFB is measured client-side from request start to the first non-empty
  audio chunk from `/tts/stream`.
- RPS is successful requests divided by phase wall-clock time.

Observed mixed-load result on 2026-05-13:

| Concurrency | All RPS | ASR p95 latency | TTS p95 latency | TTS p95 TTFB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.681 | 0.069s | 1.408s | 0.277s |
| 2 | 5.870 | 0.058s | 1.265s | 0.154s |
| 4 | 9.781 | 0.095s | 1.467s | 0.198s |
| 8 | 12.349 | 0.182s | 2.161s | 0.362s |
| 16 | 11.884 | 0.351s | 5.442s | 0.675s |
| 32 | 12.595 | 0.799s | 8.367s | 1.447s |
| 64 | 12.391 | 1.880s | 14.970s | 2.852s |
| 96 | 12.319 | 3.394s | 19.523s | 4.701s |
| 128 | 12.449 | 6.734s | 19.810s | 8.605s |
| 160 | 12.289 | 10.393s | 20.068s | 11.977s |

Throughput plateaued around 12.3 requests/second. Severe degradation was first
observed at mixed concurrency 160, where TTS p95 TTFB exceeded 10 seconds and
ASR p95 latency rose to about 10 seconds.
