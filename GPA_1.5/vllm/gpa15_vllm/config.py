from __future__ import annotations

from typing import Any

from transformers import Qwen2Config, WhisperConfig


class ArkasrConfig(Qwen2Config):
    model_type = "arkasr"
    is_composition = True

    def __init__(
        self,
        whisper_config: dict[str, Any] | WhisperConfig | None = None,
        adapter_type: str = "mlp",
        merge_factor: int = 4,
        spec_aug: bool = False,
        use_rope: bool = True,
        max_whisper_length: int = 1500,
        mlp_adapter_act: str = "gelu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(whisper_config, dict):
            self.whisper_config = WhisperConfig(**whisper_config)
        elif isinstance(whisper_config, WhisperConfig):
            self.whisper_config = whisper_config
        else:
            self.whisper_config = WhisperConfig()

        self.adapter_type = adapter_type
        self.merge_factor = int(merge_factor)
        self.spec_aug = bool(spec_aug)
        self.use_rope = bool(use_rope)
        self.max_whisper_length = int(max_whisper_length)
        self.mlp_adapter_act = mlp_adapter_act

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["whisper_config"] = self.whisper_config.to_dict()
        return output

