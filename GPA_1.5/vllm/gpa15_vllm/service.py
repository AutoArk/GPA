from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
import soundfile as sf

from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs

from models.bicodec_tokenizer.spark_detokenizer import SparkDeTokenizer
from models.bicodec_tokenizer.spark_tokenizer import SparkTokenizer

_GPA15_DIR = Path(__file__).resolve().parents[2]
if str(_GPA15_DIR) not in sys.path:
    sys.path.append(str(_GPA15_DIR))
from tts_han_char_tokenizer import encode_tts_content_text

from .local_hf import load_local_processor
from .register import register

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

logger = logging.getLogger(__name__)

TRAILING_ROLE_TOKEN_RE = re.compile(r"(?:\\s*<\\|(user|assistant|system)\\|>)+\\s*$")
BICODEC_SEMANTIC_RE = re.compile(r"<\|bicodec_semantic_(\d+)\|>")


def normalize_asr_text(text: str) -> str:
    return TRAILING_ROLE_TOKEN_RE.sub("", text.strip()).strip()


def build_conversation(audio_path: str, begin_time: float, end_time: float) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": audio_path,
                    "begin_time": begin_time,
                    "end_time": end_time,
                },
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        }
    ]


def build_prompt_text() -> str:
    return (
        "<|user|><|begin_of_audio|><|audio|><|end_of_audio|>"
        "Please transcribe this audio.<|assistant|>"
    )


def _as_token_list(tokens: torch.Tensor) -> list[int]:
    token_list = tokens.detach().cpu().long().reshape(-1).tolist()
    return [int(token) for token in token_list]


def build_tts_prompt_text(text: str, global_tokens: torch.Tensor) -> str:
    return (
        "<|user|>"
        "Given the reference audio, synthesize speech for the following text in the same voice."
        + "<|start_global_token|>"
        + "".join(f"<|bicodec_global_{token}|>" for token in _as_token_list(global_tokens))
        + "<|end_global_token|>"
        "<|start_content|>"
        f"{text}"
        "<|end_content|>"
        "<|assistant|>"
    )


def build_tts_prompt_ids(text: str, global_tokens: torch.Tensor) -> list[int]:
    tokenizer = state.processor.tokenizer
    prompt_text = build_tts_prompt_text(text, global_tokens)
    start_tag = "<|start_content|>"
    end_tag = "<|end_content|>"
    start_idx = prompt_text.index(start_tag) + len(start_tag)
    end_idx = prompt_text.index(end_tag, start_idx)

    prefix_text = prompt_text[:start_idx]
    content_text = prompt_text[start_idx:end_idx]
    suffix_text = prompt_text[end_idx:]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    content_ids = encode_tts_content_text(tokenizer, content_text)["input_ids"]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    return [int(token_id) for token_id in prefix_ids + content_ids + suffix_ids]


def extract_semantic_tokens(text: str) -> list[int]:
    return [int(token) for token in BICODEC_SEMANTIC_RE.findall(text)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPA 1.5 vLLM ASR service")
    parser.add_argument("--model", default=os.getenv("GPA_MODEL_PATH", "/data2/model/AutoArk/GPA-v1_5-0_6B"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--dtype", default=os.getenv("VLLM_DTYPE", "bfloat16"))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.getenv("VLLM_MAX_NUM_SEQS", "16")))
    parser.add_argument("--max-num-batched-tokens", type=int, default=int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "16384")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--enforce-eager", action="store_true", default=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1")
    parser.add_argument("--disable-log-stats", action="store_true", default=os.getenv("VLLM_DISABLE_LOG_STATS", "0") == "1")
    parser.add_argument("--max-audio-seconds", type=int, default=int(os.getenv("GPA_MAX_AUDIO_SECONDS", "30")))
    parser.add_argument("--sampling-rate", type=int, default=int(os.getenv("GPA_SAMPLING_RATE", "16000")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("GPA_MAX_NEW_TOKENS", "256")))
    parser.add_argument("--bicodec-model", default=os.getenv("GPA_BICODEC_PATH", "/data2/model/AutoArk/GPA/BiCodec"))
    parser.add_argument("--tts-device", default=os.getenv("GPA_TTS_DEVICE", "cuda"))
    parser.add_argument("--tts-max-new-tokens", type=int, default=int(os.getenv("GPA_TTS_MAX_NEW_TOKENS", "1024")))
    return parser.parse_args()


class AppState:
    args: argparse.Namespace
    processor: Any
    bicodec_tokenizer: SparkTokenizer
    bicodec_detokenizer: SparkDeTokenizer
    engine: AsyncLLMEngine


state = AppState()
processor_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    register()
    state.args = args
    state.processor = load_local_processor(args.model)
    state.bicodec_tokenizer = SparkTokenizer(
        model_path=args.bicodec_model,
        device=args.tts_device,
        attn_implementation="eager",
    )
    state.bicodec_detokenizer = SparkDeTokenizer(
        model_path=args.bicodec_model,
        device=args.tts_device,
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        config_format="arkasr",
        tokenizer_mode="hf",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        disable_log_stats=args.disable_log_stats,
        enable_log_requests=True,
        limit_mm_per_prompt={"audio": 1},
        enable_mm_embeds=True,
        mm_processor_kwargs={
            "audio_max_length": args.max_audio_seconds * args.sampling_rate,
            "audio_padding": "longest",
            "sampling_rate": args.sampling_rate,
            "text_kwargs": {"padding": "longest"},
        },
    )
    state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield


app = FastAPI(title="GPA 1.5 vLLM ASR", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def prepare_prompt(
    audio_path: str,
    begin_time: float,
    end_time: float,
) -> TokensPrompt:
    args = state.args
    inputs = state.processor.apply_chat_template(
        build_conversation(audio_path, begin_time, end_time),
        return_tensors="pt",
        sampling_rate=args.sampling_rate,
        audio_padding="longest",
        add_generation_prompt=True,
        text_kwargs={"padding": "longest"},
        audio_max_length=args.max_audio_seconds * args.sampling_rate,
    )
    if "audios" not in inputs:
        raise RuntimeError(f"Processor output missing audios, keys={list(inputs.keys())}")

    input_ids = state.processor.tokenizer.encode(
        build_prompt_text(),
        add_special_tokens=False,
    )
    audios = inputs["audios"]
    if torch.is_tensor(audios) and audios.ndim == 2:
        audios = audios.unsqueeze(0)

    return TokensPrompt(
        prompt_token_ids=input_ids,
        multi_modal_data={"audio": {"audios": audios}},
    )


async def run_generation(prompt: TokensPrompt, max_new_tokens: int) -> str:
    request_id = f"gpa15-{uuid.uuid4().hex}"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop_token_ids=[151665],
        skip_special_tokens=False,
    )
    final_output = None
    async for output in state.engine.generate(prompt, sampling_params, request_id):
        final_output = output
    if final_output is None or not final_output.outputs:
        return ""
    return normalize_asr_text(final_output.outputs[0].text)


async def run_tts_generation(prompt_token_ids: list[int], max_new_tokens: int) -> str:
    tokenizer = state.processor.tokenizer
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    request_id = f"gpa15-tts-{uuid.uuid4().hex}"
    sampling_params = SamplingParams(
        temperature=0.3,
        repetition_penalty=1.1,
        max_tokens=max_new_tokens,
        stop_token_ids=[im_end_id],
        skip_special_tokens=False,
    )
    final_output = None
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    async for output in state.engine.generate(prompt, sampling_params, request_id):
        final_output = output
    if final_output is None or not final_output.outputs:
        return ""
    return final_output.outputs[0].text


def prepare_tts_request(
    text: str,
    ref_audio_path: str,
) -> tuple[list[int], torch.Tensor]:
    tokenizer_result = state.bicodec_tokenizer.tokenize([ref_audio_path])
    global_tokens = tokenizer_result["global_tokens"]
    prompt_token_ids = build_tts_prompt_ids(text, global_tokens)
    return prompt_token_ids, global_tokens


def detokenize_tts(global_tokens: torch.Tensor, semantic_tokens: list[int]) -> bytes:
    if not semantic_tokens:
        raise RuntimeError("TTS generation produced no bicodec semantic tokens")
    req = {
        "global_tokens": global_tokens,
        "semantic_tokens": torch.tensor(semantic_tokens, dtype=torch.long).unsqueeze(0),
    }
    audio = state.bicodec_detokenizer.detokenize(**req)
    wav = audio.detach().cpu().float().squeeze().numpy()
    if wav.size > 0:
        wav = wav - wav.mean()

    bio = io.BytesIO()
    sf.write(bio, wav, state.args.sampling_rate, format="WAV")
    return bio.getvalue()


def detokenize_tts_pcm(global_tokens: torch.Tensor, semantic_tokens: list[int]) -> bytes:
    if not semantic_tokens:
        return b""
    req = {
        "global_tokens": global_tokens,
        "semantic_tokens": torch.tensor(semantic_tokens, dtype=torch.long).unsqueeze(0),
    }
    audio = state.bicodec_detokenizer.detokenize(**req)
    wav = audio.detach().cpu().float().squeeze().numpy()
    if wav.size > 0:
        wav = wav - wav.mean()
    wav = wav.clip(-1.0, 1.0)
    return (wav * 32767.0).astype("<i2", copy=False).tobytes()


async def stream_tts_pcm(
    text: str,
    ref_audio_path: str,
    max_new_tokens: int,
    chunk_semantic_tokens: int,
):
    async with processor_lock:
        prompt_token_ids, global_tokens = await asyncio.to_thread(
            prepare_tts_request,
            text,
            ref_audio_path,
        )

    tokenizer = state.processor.tokenizer
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    request_id = f"gpa15-tts-stream-{uuid.uuid4().hex}"
    sampling_params = SamplingParams(
        temperature=0.3,
        repetition_penalty=1.1,
        max_tokens=max_new_tokens,
        stop_token_ids=[im_end_id],
        skip_special_tokens=False,
    )
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

    seen_semantic_tokens = 0
    pending_tokens: list[int] = []
    yielded_any = False

    async for output in state.engine.generate(prompt, sampling_params, request_id):
        if not output.outputs:
            continue
        semantic_tokens = extract_semantic_tokens(output.outputs[0].text)
        new_tokens = semantic_tokens[seen_semantic_tokens:]
        if new_tokens:
            pending_tokens.extend(new_tokens)
            seen_semantic_tokens += len(new_tokens)

        while len(pending_tokens) >= chunk_semantic_tokens:
            chunk = pending_tokens[:chunk_semantic_tokens]
            del pending_tokens[:chunk_semantic_tokens]
            async with processor_lock:
                audio_bytes = await asyncio.to_thread(
                    detokenize_tts_pcm,
                    global_tokens,
                    chunk,
                )
            if audio_bytes:
                yielded_any = True
                yield audio_bytes

    if pending_tokens:
        async with processor_lock:
            audio_bytes = await asyncio.to_thread(
                detokenize_tts_pcm,
                global_tokens,
                pending_tokens,
            )
        if audio_bytes:
            yielded_any = True
            yield audio_bytes

    if not yielded_any:
        logger.warning("TTS stream produced no audio chunks")


@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    begin_time: float = Form(-1),
    end_time: float = Form(-1),
    max_new_tokens: int | None = Form(None),
) -> JSONResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    started = time.perf_counter()
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        async with processor_lock:
            prompt = await asyncio.to_thread(
                prepare_prompt,
                tmp_path,
                begin_time,
                end_time,
            )
        text = await run_generation(
            prompt,
            max_new_tokens=max_new_tokens or state.args.max_new_tokens,
        )
        latency = time.perf_counter() - started
        return JSONResponse(
            {
                "text": text,
                "latency_s": latency,
                "prompt_tokens": len(prompt["prompt_token_ids"]),
            }
        )
    except Exception as exc:
        logger.exception("ASR request failed")
        detail = f"{exc.__class__.__name__}: {exc}"
        if exc.__cause__ is not None:
            detail += f"; cause={exc.__cause__.__class__.__name__}: {exc.__cause__}"
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    response_format: str | None = Form(None),
    temperature: float | None = Form(None),
) -> JSONResponse:
    del model, response_format, temperature
    return await asr(file=file, begin_time=-1, end_time=-1, max_new_tokens=None)


@app.post("/tts")
async def tts(
    text: str = Form(...),
    ref_file: UploadFile = File(...),
    max_new_tokens: int | None = Form(None),
) -> Response:
    suffix = Path(ref_file.filename or "ref.wav").suffix or ".wav"
    started = time.perf_counter()
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await ref_file.read()
            tmp.write(content)

        async with processor_lock:
            prompt_token_ids, global_tokens = await asyncio.to_thread(
                prepare_tts_request,
                text,
                tmp_path,
            )
        generated_text = await run_tts_generation(
            prompt_token_ids,
            max_new_tokens=max_new_tokens or state.args.tts_max_new_tokens,
        )
        semantic_tokens = extract_semantic_tokens(generated_text)
        async with processor_lock:
            audio_bytes = await asyncio.to_thread(
                detokenize_tts,
                global_tokens,
                semantic_tokens,
            )

        latency = time.perf_counter() - started
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-GPA-Latency-S": f"{latency:.6f}",
                "X-GPA-Prompt-Tokens": str(len(prompt_token_ids)),
                "X-GPA-Semantic-Tokens": str(len(semantic_tokens)),
            },
        )
    except Exception as exc:
        logger.exception("TTS request failed")
        detail = f"{exc.__class__.__name__}: {exc}"
        if exc.__cause__ is not None:
            detail += f"; cause={exc.__cause__.__class__.__name__}: {exc.__cause__}"
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(...),
    ref_file: UploadFile = File(...),
    max_new_tokens: int | None = Form(None),
    chunk_semantic_tokens: int = Form(32),
) -> StreamingResponse:
    suffix = Path(ref_file.filename or "ref.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        content = await ref_file.read()
        tmp.write(content)
        tmp.close()
    except Exception:
        tmp.close()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    async def body():
        try:
            async for chunk in stream_tts_pcm(
                text=text,
                ref_audio_path=tmp_path,
                max_new_tokens=max_new_tokens or state.args.tts_max_new_tokens,
                chunk_semantic_tokens=max(int(chunk_semantic_tokens), 1),
            ):
                yield chunk
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        body(),
        media_type="audio/L16; rate=16000; channels=1",
        headers={"X-GPA-Audio-Format": "pcm_s16le"},
    )


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run("gpa15_vllm.service:app", host=args.host, port=args.port, factory=False)


if __name__ == "__main__":
    main()
