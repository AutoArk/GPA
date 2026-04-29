import os
import tempfile
import threading
import time
import uuid
from math import gcd
from pathlib import Path

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import soundfile as sf
from scipy.signal import resample_poly
from typing import Optional

from infer_ark_audio_onnx import ArkAudioOnnxRuntime
from voice_registry import VoiceRegistry


APP_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET_ROOT = (APP_DIR.parent.parent / "GPA-v1.5-HF" / "GPA-v1.5-onnx-runtime").resolve()
LEGACY_ASSET_ROOT = (APP_DIR.parent.parent / "GPA-v1.5-HF" / "GPA_v1.5_onnx_runtime").resolve()
AUTO_ASSET_ROOT = DEFAULT_ASSET_ROOT if DEFAULT_ASSET_ROOT.exists() or not LEGACY_ASSET_ROOT.exists() else LEGACY_ASSET_ROOT
ASSET_ROOT = Path(os.environ.get("ARK_AUDIO_RUNTIME_ASSET_ROOT", str(AUTO_ASSET_ROOT))).expanduser().resolve()
MODEL_DIR = ASSET_ROOT / "model"
BUILD_DIR = ASSET_ROOT / "build"
DEFAULT_TOKENIZER_MODEL_DIR = ASSET_ROOT / "voice" / "spark_tokenizer_model"
OUTPUT_DIR = APP_DIR / "outputs"
STATIC_DIR = APP_DIR / "static"
SAMPLES_DIR = APP_DIR / "samples"
VOICE_DIR = APP_DIR / "voices"
DEFAULT_REFERENCE_TOKEN_PATH = MODEL_DIR / "reference" / "default_global_tokens.npy"
TOKENIZER_MODEL_DIR = Path(
    os.environ.get(
        "ARK_AUDIO_TOKENIZER_MODEL_DIR",
        str(DEFAULT_TOKENIZER_MODEL_DIR),
    )
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GPA ONNX Runtime")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

process = psutil.Process()
service_peak_rss = process.memory_info().rss
last_request_peak_rss = service_peak_rss
service_lock = threading.Lock()
runtime: Optional[ArkAudioOnnxRuntime] = None
registry = VoiceRegistry(VOICE_DIR)
registration_tokenizer_loaded = False


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    voice_name: str = Field("default", min_length=1, max_length=120)
    decoder_precision: str = Field("fp16", pattern="^(int8|fp16)$")
    main_model_precision: str = Field("fp16", pattern="^(fp32|fp16|int8|int4)$")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    repetition_penalty: float = Field(1.5, ge=0.5, le=3.0)
    max_new_tokens: int = Field(512, ge=32, le=2048)


class PathRegistrationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    audio_path: str = Field(..., min_length=1)
    overwrite: bool = False


class OpenAISpeechRequest(BaseModel):
    model: str = Field("tts-1", min_length=1)
    input: str = Field(..., min_length=1, max_length=500)
    voice: str = Field("default", min_length=1, max_length=120)
    response_format: str = Field("pcm", min_length=1, max_length=20)
    speed: Optional[float] = None
    instructions: Optional[str] = None


def preferred_main_model_precision() -> str:
    if runtime is not None:
        return runtime.default_main_model_precision()
    return "fp16"


def preferred_decoder_precision() -> str:
    if runtime is not None:
        detok_name = str(runtime.manifest.get("detokenizer_default", "spark_detokenizer_fp16.onnx"))
        return "int8" if "int8" in detok_name else "fp16"
    return "fp16"


def update_service_peak(peak_candidate: Optional[int] = None) -> None:
    global service_peak_rss
    current = process.memory_info().rss
    if peak_candidate is None:
        service_peak_rss = max(service_peak_rss, current)
    else:
        service_peak_rss = max(service_peak_rss, current, int(peak_candidate))


def memory_snapshot() -> dict:
    update_service_peak()
    available_main_model_precisions = []
    default_main_model_precision = None
    tts_text_tokenizer_mode = None
    if runtime is not None:
        available_main_model_precisions = runtime.available_main_model_precisions()
        default_main_model_precision = runtime.default_main_model_precision()
        tts_text_tokenizer_mode = runtime.tts_text_tokenizer_mode()
    return {
        "rss_bytes": process.memory_info().rss,
        "peak_rss_bytes": service_peak_rss,
        "last_request_peak_rss_bytes": last_request_peak_rss,
        "runtime_loaded": runtime is not None,
        "model_dir_present": MODEL_DIR.exists(),
        "build_dir_present": BUILD_DIR.exists(),
        "registered_voice_count": len(registry.list_voices()),
        "registration_tokenizer_loaded": registration_tokenizer_loaded,
        "tokenizer_model_dir_present": TOKENIZER_MODEL_DIR.exists(),
        "available_main_model_precisions": available_main_model_precisions,
        "default_main_model_precision": default_main_model_precision,
        "tts_text_tokenizer_mode": tts_text_tokenizer_mode,
    }


def list_voices_payload() -> list[dict]:
    voices = []
    for voice in registry.list_voices():
        voices.append(
            {
                "name": voice["name"],
                "created_at": voice.get("created_at"),
                "updated_at": voice.get("updated_at"),
                "source_kind": voice.get("source_kind"),
                "source_label": voice.get("source_label"),
                "is_default": bool(voice.get("is_default", False)),
            }
        )
    return voices


def list_openai_voice_payload() -> list[dict]:
    return [{"id": voice["name"], "name": voice["name"]} for voice in list_voices_payload()]


def list_models_payload() -> list[dict]:
    created = int(time.time())
    return [
        {"id": "tts-1", "object": "model", "created": created, "owned_by": "ark_audio_onnx_runtime"},
        {"id": "whisper-1", "object": "model", "created": created, "owned_by": "ark_audio_onnx_runtime"},
    ]


def choose_voice_name(requested_voice: str) -> str:
    voices = list_voices_payload()
    if not voices:
        raise HTTPException(status_code=503, detail="No voices are registered.")

    requested_voice = (requested_voice or "").strip()
    available_names = [voice["name"] for voice in voices]
    default_voice = next((voice["name"] for voice in voices if voice.get("is_default")), available_names[0])

    if requested_voice in available_names:
        return requested_voice
    if "default" in available_names:
        return "default"
    return default_voice


PCM_SAMPLE_RATE = 24000


def wav_file_to_pcm_bytes(wav_path: Path) -> bytes:
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    if sample_rate != PCM_SAMPLE_RATE:
        factor = gcd(int(sample_rate), PCM_SAMPLE_RATE)
        up = PCM_SAMPLE_RATE // factor
        down = int(sample_rate) // factor
        audio = resample_poly(audio, up, down).astype("float32", copy=False)
    pcm = (audio.clip(-1.0, 1.0) * 32767.0).astype("int16")
    return pcm.tobytes()


def wav_file_bytes(wav_path: Path) -> bytes:
    return wav_path.read_bytes()


def build_openai_error(message: str, status_code: int = 400, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


class RequestMemorySampler:
    def __init__(self, sample_interval_s: float = 0.05):
        self.sample_interval_s = sample_interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss_bytes = process.memory_info().rss

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            self.peak_rss_bytes = max(self.peak_rss_bytes, process.memory_info().rss)

    def start(self) -> None:
        self.peak_rss_bytes = process.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self.peak_rss_bytes = max(self.peak_rss_bytes, process.memory_info().rss)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_rss_bytes


def ensure_default_voice() -> dict:
    return registry.ensure_default_from_reference(reference_token_path=DEFAULT_REFERENCE_TOKEN_PATH, default_name="default")


def require_runtime() -> ArkAudioOnnxRuntime:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded.")
    return runtime


def require_registration_assets() -> None:
    if not TOKENIZER_MODEL_DIR.exists():
        raise HTTPException(status_code=503, detail=f"Tokenizer model dir not found: {TOKENIZER_MODEL_DIR}")


def finalize_request_peak(peak_rss_bytes: int) -> None:
    global last_request_peak_rss
    last_request_peak_rss = max(int(peak_rss_bytes), process.memory_info().rss)
    update_service_peak(last_request_peak_rss)


def run_registration(*, name: str, audio_path: Path, source_kind: str, source_label: str, overwrite: bool) -> dict:
    global registration_tokenizer_loaded
    require_registration_assets()
    registration_tokenizer_loaded = True
    try:
        from voice_registration import register_voice_from_audio

        return register_voice_from_audio(
            tokenizer_model_dir=TOKENIZER_MODEL_DIR,
            registry=registry,
            name=name,
            audio_path=audio_path,
            source_kind=source_kind,
            source_label=source_label,
            overwrite=overwrite,
            device="cpu",
        )
    finally:
        registration_tokenizer_loaded = False
        update_service_peak()


@app.on_event("startup")
def startup_event() -> None:
    global runtime
    ensure_default_voice()
    runtime = ArkAudioOnnxRuntime(ASSET_ROOT)
    update_service_peak()


@app.on_event("shutdown")
def shutdown_event() -> None:
    global runtime
    runtime = None


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "memory": memory_snapshot(), "voices": list_voices_payload()}


@app.get("/api/memory")
def memory() -> dict:
    return memory_snapshot()


@app.get("/api/voices")
def voices() -> dict:
    return {"ok": True, "voices": list_voices_payload(), "service_memory": memory_snapshot()}


@app.get("/v1/models")
def openai_models() -> dict:
    return {"object": "list", "data": list_models_payload()}


@app.get("/v1/audio/voices")
def openai_voices() -> dict:
    return {"voices": list_openai_voice_payload()}


@app.post("/api/voices/register-path")
def register_path(request: PathRegistrationRequest) -> dict:
    audio_path = Path(request.audio_path).expanduser().resolve()
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=400, detail=f"Audio file not found: {audio_path}")

    with service_lock:
        try:
            result = run_registration(
                name=request.name,
                audio_path=audio_path,
                source_kind="path",
                source_label=audio_path.name,
                overwrite=request.overwrite,
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    finalize_request_peak(result["peak_rss_bytes"])
    return {
        "ok": True,
        "voice": result["voice"],
        "voices": list_voices_payload(),
        "result": result,
        "service_memory": memory_snapshot(),
    }


@app.post("/api/voices/register-upload")
async def register_upload(
    name: str = Form(...),
    overwrite: bool = Form(False),
    audio: UploadFile = File(...),
) -> dict:
    suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ark_voice_", suffix=suffix, delete=False) as handle:
            tmp_path = Path(handle.name)
            handle.write(await audio.read())

        with service_lock:
            try:
                result = run_registration(
                    name=name,
                    audio_path=tmp_path,
                    source_kind="upload",
                    source_label=audio.filename or tmp_path.name,
                    overwrite=overwrite,
                )
            except FileExistsError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        finalize_request_peak(result["peak_rss_bytes"])
        return {
            "ok": True,
            "voice": result["voice"],
            "voices": list_voices_payload(),
            "result": result,
            "service_memory": memory_snapshot(),
        }
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/api/tts")
def tts(request: TTSRequest) -> dict:
    runtime_obj = require_runtime()
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        voice = registry.require_voice(request.voice_name.strip())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    output_name = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    output_path = OUTPUT_DIR / output_name
    global_token_path = registry.global_token_path_for_voice(voice)
    sampler = RequestMemorySampler()
    sampler.start()
    with service_lock:
        try:
            sem_ids, gen_text = runtime_obj.run_tts(
                text=text,
                output_path=output_path,
                global_token_path=global_token_path,
                max_new_tokens=request.max_new_tokens,
                decoder_precision=request.decoder_precision,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                main_model_precision=request.main_model_precision,
            )
        finally:
            finalize_request_peak(sampler.stop())

    return {
        "ok": True,
        "voice": voice,
        "audio_url": f"/outputs/{output_name}",
        "semantic_count": len(sem_ids),
        "generated_text_preview": gen_text[:500],
        "main_model_precision": request.main_model_precision,
        "decoder_precision": request.decoder_precision,
        "temperature": request.temperature,
        "repetition_penalty": request.repetition_penalty,
        "request_peak_rss_bytes": last_request_peak_rss,
        "service_memory": memory_snapshot(),
    }


@app.post("/v1/audio/speech")
def openai_speech(request: OpenAISpeechRequest):
    if request.model != "tts-1":
        return build_openai_error(f"Unsupported TTS model: {request.model}", status_code=400)

    response_format = request.response_format.strip().lower()
    if response_format not in {"wav", "pcm"}:
        return build_openai_error(
            f"Unsupported response_format '{request.response_format}'. Supported formats: wav, pcm",
            status_code=400,
        )

    runtime_obj = require_runtime()
    text = request.input.strip()
    if not text:
        return build_openai_error("Input must not be empty.", status_code=400)

    voice_name = choose_voice_name(request.voice)
    output_name = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    output_path = OUTPUT_DIR / output_name
    global_token_path = registry.global_token_path_for_voice(registry.require_voice(voice_name))
    sampler = RequestMemorySampler()
    sampler.start()
    with service_lock:
        try:
            runtime_obj.run_tts(
                text=text,
                output_path=output_path,
                global_token_path=global_token_path,
                max_new_tokens=512,
                decoder_precision=preferred_decoder_precision(),
                temperature=0.1,
                repetition_penalty=1.2,
                main_model_precision=preferred_main_model_precision(),
            )
        except Exception as exc:
            return build_openai_error(str(exc), status_code=500, error_type="server_error")
        finally:
            finalize_request_peak(sampler.stop())

    if response_format == "wav":
        content = wav_file_bytes(output_path)
        media_type = "audio/wav"
    else:
        content = wav_file_to_pcm_bytes(output_path)
        media_type = "application/octet-stream"

    return Response(content=content, media_type=media_type)


@app.post("/api/asr")
async def asr(
    audio: UploadFile = File(...),
    begin_time: float = Form(-1),
    end_time: float = Form(-1),
    max_audio_seconds: int = Form(30),
    max_new_tokens: int = Form(256),
    main_model_precision: str = Form("fp16"),
) -> dict:
    runtime_obj = require_runtime()
    suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    sampler = RequestMemorySampler()
    sampler.start()
    with service_lock:
        try:
            text = runtime_obj.transcribe_audio_bytes(
                audio_bytes=audio_bytes,
                suffix=suffix,
                begin_time=begin_time,
                end_time=end_time,
                max_audio_seconds=max_audio_seconds,
                max_new_tokens=max_new_tokens,
                main_model_precision=main_model_precision,
            )
        finally:
            finalize_request_peak(sampler.stop())

    return {
        "ok": True,
        "text": text,
        "filename": audio.filename or f"upload{suffix}",
        "main_model_precision": main_model_precision,
        "request_peak_rss_bytes": last_request_peak_rss,
        "service_memory": memory_snapshot(),
    }


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("json"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
):
    del language
    del prompt

    if model != "whisper-1":
        return build_openai_error(f"Unsupported STT model: {model}", status_code=400)

    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    audio_bytes = await file.read()
    if not audio_bytes:
        return build_openai_error("Uploaded audio is empty.", status_code=400)

    runtime_obj = require_runtime()
    sampler = RequestMemorySampler()
    sampler.start()
    with service_lock:
        try:
            text = runtime_obj.transcribe_audio_bytes(
                audio_bytes=audio_bytes,
                suffix=suffix,
                begin_time=-1,
                end_time=-1,
                max_audio_seconds=30,
                max_new_tokens=256,
                main_model_precision=preferred_main_model_precision(),
            )
        except Exception as exc:
            return build_openai_error(str(exc), status_code=500, error_type="server_error")
        finally:
            finalize_request_peak(sampler.stop())

    if response_format == "text":
        return PlainTextResponse(text)
    return {"text": text}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
