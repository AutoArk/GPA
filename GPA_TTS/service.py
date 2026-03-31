import tempfile
import threading
import time
import uuid
from pathlib import Path

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tts_realtime_int8_light import PackagedTTSRuntime
from voice_registration import register_voice_from_audio
from voice_registry import VoiceRegistry


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
OUTPUT_DIR = APP_DIR / "outputs"
STATIC_DIR = APP_DIR / "static"
VOICE_DIR = APP_DIR / "voices"
TOKENIZER_MODEL_DIR = APP_DIR / "voice" / "spark_tokenizer_model"
DEFAULT_REFERENCE_TOKEN_PATH = MODEL_DIR / "reference" / "038142_global_tokens.npy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GPA TTS INT8 Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

process = psutil.Process()
service_peak_rss = process.memory_info().rss
service_lock = threading.Lock()
runtime: PackagedTTSRuntime | None = None
registry = VoiceRegistry(VOICE_DIR)
registration_tokenizer_loaded = False


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    voice_name: str = Field("default", min_length=1, max_length=120)
    max_new_tokens: int = Field(512, ge=32, le=2048)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    repetition_penalty: float = Field(1.2, ge=0.5, le=3.0)
    do_sample: bool = False


class PathRegistrationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    audio_path: str = Field(..., min_length=1)
    overwrite: bool = False


def update_service_peak(peak_candidate: int | None = None) -> None:
    global service_peak_rss
    current = process.memory_info().rss
    if peak_candidate is None:
        service_peak_rss = max(service_peak_rss, current)
    else:
        service_peak_rss = max(service_peak_rss, current, int(peak_candidate))


def memory_snapshot() -> dict:
    update_service_peak()
    return {
        "rss_bytes": process.memory_info().rss,
        "peak_rss_bytes": service_peak_rss,
        "inference_models_resident": runtime is not None,
        "registration_tokenizer_loaded": registration_tokenizer_loaded,
        "registered_voice_count": len(registry.list_voices()),
    }


def ensure_default_voice() -> dict:
    return registry.ensure_default_from_reference(reference_token_path=DEFAULT_REFERENCE_TOKEN_PATH, default_name="default")


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


def run_registration(*, name: str, audio_path: Path, source_kind: str, source_label: str, overwrite: bool) -> dict:
    global registration_tokenizer_loaded
    registration_tokenizer_loaded = True
    try:
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
    default_voice = ensure_default_voice()
    if runtime is None:
        runtime = PackagedTTSRuntime(
            bundle_dir=MODEL_DIR,
            default_global_token_path=registry.global_token_path_for_voice(default_voice),
        )
    update_service_peak()


@app.on_event("shutdown")
def shutdown_event() -> None:
    global runtime
    if runtime is not None:
        runtime.close()
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
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        with tempfile.NamedTemporaryFile(prefix="gpa_voice_", suffix=suffix, delete=False) as handle:
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
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded.")

    try:
        voice = registry.require_voice(request.voice_name.strip())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    output_name = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    output_path = OUTPUT_DIR / output_name
    global_token_path = registry.global_token_path_for_voice(voice)

    with service_lock:
        result = runtime.synthesize_to_file(
            text=text,
            output_path=output_path,
            global_token_path=global_token_path,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        update_service_peak(result["peak_rss_bytes"])

    return {
        "ok": True,
        "voice": voice,
        "audio_url": f"/outputs/{output_name}",
        "result": result,
        "service_memory": memory_snapshot(),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
