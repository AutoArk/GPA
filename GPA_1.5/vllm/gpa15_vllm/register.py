from __future__ import annotations

import os

from transformers import AutoConfig

from vllm import ModelRegistry

from .config import ArkasrConfig
from . import config_parser as _config_parser  # noqa: F401

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    AutoConfig.register("arkasr", ArkasrConfig, exist_ok=True)
    from .modeling import ArkasrForConditionalGeneration

    ModelRegistry.register_model(
        "ArkasrForConditionalGeneration",
        ArkasrForConditionalGeneration,
    )
    os.environ.setdefault("VLLM_CONFIG_FORMAT", "arkasr")
    _REGISTERED = True


register()
