# -*- coding: utf-8 -*-
import argparse
import uvicorn
import torch
import os
import uuid
import io
import wave
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from typing import Optional
from contextlib import asynccontextmanager

# Import your Engine
from gpa_engine import ArkGPAEngine

# ======================== 参数解析 ========================

def parse_args():
    parser = argparse.ArgumentParser(description="ArkGPA Multimodal Voice Server (Non-Streaming)")

    # Model path related
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data3/gpa_ckpt/gpa_0_3_b_577000",
        help="LLM model path",
    )
    parser.add_argument(
        "--bicodec_path",
        type=str,
        default="/data3/ark_tts_v1",
        help="Bicodec audio tokenizer path",
    )
    parser.add_argument("--glm_path", type=str, default="/data/yumu/model/glm-4-voice-tokenizer", help="GLM speech tokenizer path")
    parser.add_argument(
        "--text_tokenizer_path",
        type=str,
        default="/data3/gpa_ckpt/gpa_0_3_b_577000",
        help="Text tokenizer path",
    )

    # Runtime environment related
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--llm_gpu_memory_utilization", type=float, default=0.1, help="gpu utilization for vllm")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu). If None, auto-detect.")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "torch", "sglang"], help="Inference backend")

    return parser.parse_args()

# Global variables
engine: Optional[ArkGPAEngine] = None
args = parse_args()

# ======================== Lifespan Handler ========================
TEMP_DIR ="tmp_dir"
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Unified management of engine startup and shutdown
    """
    global engine
    if not os.path.exists(TEMP_DIR):
        print(f"Creating temporary directory: {os.path.abspath(TEMP_DIR)}")
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Configuration ---")
    print(f"Model Path: {args.model_path}")
    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"Temp Dir: {os.path.abspath(TEMP_DIR)}")
    print(f"----------------------")
    
    print("Initializing ArkGPA Engine, please wait...")
    engine = ArkGPAEngine(
        model_path=args.model_path,
        bicodec_audio_tokenizer_path=args.bicodec_path,
        glm_speech_tokenizer_path=args.glm_path,
        text_tokenizer_path=args.text_tokenizer_path,
        llm_device=device,
        llm_gpu_memory_utilization=args.llm_gpu_memory_utilization,
        backend=args.backend
    )
    print("Engine Initialized successfully.")
    
    yield
    
    print("Shutting down...")
    if engine:
        del engine

# Create application
app = FastAPI(title="ArkGPA Multimodal Voice Server", lifespan=lifespan)

# ======================== Helper Functions ========================

async def save_temp_audio(file: UploadFile):
    """保存上传的文件到临时路径并返回路径"""
    content = await file.read()

    filename = file.filename if file.filename else "temp.wav"
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = ".wav"

    unique_filename = f"temp_{uuid.uuid4().hex}{ext}"
    full_path = os.path.join(TEMP_DIR, unique_filename)

    with open(full_path, "wb") as f:
        f.write(content)

    return full_path

def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    将 numpy 数组 (int16) 转换为 WAV 格式的字节流
    方便客户端直接保存播放，而不需要处理 raw PCM
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return buffer.getvalue()

# ======================== Interface Implementation ========================


@app.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    top_k: Optional[int] = Form(None),
    top_p: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
):
    """
    Non-streaming Speech to Text
    Returns: JSON {"text": "..."}
    """
    audio_path = await save_temp_audio(file)
    try:
        text = await engine.stt_async(
            audio=audio_path,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        text = text.replace("<|end_content|>","")
        return JSONResponse(content={"text": text})
    finally:
        if os.path.exists(audio_path): 
            os.remove(audio_path)


@app.post("/tts/a")
async def tts_mode_a(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    top_k: Optional[int] = Form(None),
    top_p: Optional[float] = Form(None),
    temperature: Optional[float] = Form(0.2),
    repetition_penalty: Optional[float] = Form(1.2),
):
    """
    Non-streaming TTS Mode A (Zero-shot Cloning)
    Returns: WAV file bytes
    """
    ref_path = await save_temp_audio(ref_audio)
    try:
        # 调用非流式接口 tts_a_async
        # 返回的是 numpy array (int16)
        audio_data = await engine.tts_a_async(
            text=text,
            ref_audio=ref_path,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        # 转为 WAV 格式
        wav_bytes = numpy_to_wav_bytes(audio_data)

        # 返回 wav 文件流
        return Response(content=wav_bytes, media_type="audio/wav")
    finally:
        if os.path.exists(ref_path): os.remove(ref_path)


@app.post("/vc")
async def voice_conversion(
    src_audio: UploadFile = File(...),
    ref_audio: UploadFile = File(...),
    top_k: Optional[int] = Form(None),
    top_p: Optional[float] = Form(None),
    temperature: Optional[float] = Form(0.2),
    repetition_penalty: Optional[float] = Form(1.2),
):
    src_path = await save_temp_audio(src_audio)
    ref_path = await save_temp_audio(ref_audio)
    try:
        audio_data = await engine.vc_async(
            src_audio=src_path,
            ref_audio=ref_path,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        wav_bytes = numpy_to_wav_bytes(audio_data)
        return Response(content=wav_bytes, media_type="audio/wav")
    finally:
        if os.path.exists(src_path): os.remove(src_path)
        if os.path.exists(ref_path): os.remove(ref_path)


# ======================== 启动入口 ========================

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
