import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE_ROOT / "GPA-v1.5"
DEFAULT_MODEL_DIR = WORKSPACE_ROOT / "GPA-v1.5-HF" / "GPA-v1.5"
LEGACY_MODEL_DIR = WORKSPACE_ROOT / "GPA-v1.5-HF" / "GPA_v1.5"
DEFAULT_AUDIO_TOKENIZER_DIR = DEFAULT_MODEL_DIR / "spark_tokenizer_model"
LEGACY_AUDIO_TOKENIZER_DIR = LEGACY_MODEL_DIR / "spark_tokenizer_model"
MODEL_ENV_VAR = "ARK_AUDIO_HF_MODEL_DIR"
TOKENIZER_ENV_VAR = "ARK_AUDIO_TOKENIZER_MODEL_DIR"


def build_default_path_report() -> dict[str, str]:
    model_env = os.environ.get(MODEL_ENV_VAR)
    tokenizer_env = os.environ.get(TOKENIZER_ENV_VAR)
    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "model_env_var": MODEL_ENV_VAR,
        "model_env_value": model_env or "",
        "model_default": str(DEFAULT_MODEL_DIR),
        "tokenizer_env_var": TOKENIZER_ENV_VAR,
        "tokenizer_env_value": tokenizer_env or "",
        "tokenizer_default": str(DEFAULT_AUDIO_TOKENIZER_DIR),
    }


def _resolve_candidate(
    explicit: str | None,
    env_var: str,
    default_path: Path,
    legacy_default_path: Path | None = None,
) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    if legacy_default_path and not default_path.exists() and legacy_default_path.exists():
        return legacy_default_path.resolve()
    return default_path.resolve()


def resolve_model_dir(explicit: str | None = None) -> Path:
    path = _resolve_candidate(explicit, MODEL_ENV_VAR, DEFAULT_MODEL_DIR, LEGACY_MODEL_DIR)
    if not path.exists():
        raise FileNotFoundError(
            "Model directory not found. "
            f"Checked '{path}'. Provide --model-path or set {MODEL_ENV_VAR}."
        )
    return path


def resolve_audio_tokenizer_dir(explicit: str | None = None) -> Path:
    path = _resolve_candidate(
        explicit,
        TOKENIZER_ENV_VAR,
        DEFAULT_AUDIO_TOKENIZER_DIR,
        LEGACY_AUDIO_TOKENIZER_DIR,
    )
    if not path.exists():
        raise FileNotFoundError(
            "Spark tokenizer directory not found. "
            f"Checked '{path}'. Provide --audio-tokenizer-path or set {TOKENIZER_ENV_VAR}."
        )
    return path