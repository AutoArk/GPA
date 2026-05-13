from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

from vllm.transformers_utils.config import register_config_parser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .config import ArkasrConfig


@register_config_parser("arkasr")
class ArkasrConfigParser(ConfigParserBase):
    """Load ArkasrConfig without relying on HF dynamic module cache writes."""

    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, PretrainedConfig]:
        del trust_remote_code, revision, code_revision

        config_dict, _ = PretrainedConfig.get_config_dict(model, **kwargs)
        return config_dict, ArkasrConfig.from_dict(config_dict)

