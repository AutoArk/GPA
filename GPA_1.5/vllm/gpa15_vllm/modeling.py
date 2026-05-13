from __future__ import annotations

import copy
import inspect
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature, Qwen2Config, WhisperFeatureExtractor
from transformers.processing_utils import ProcessorMixin

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.model_executor.models.whisper_utils import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config

from .config import ArkasrConfig
from .local_hf import load_local_module, load_local_processor

logger = init_logger(__name__)

MultiModalDataDict: TypeAlias = dict[str, object]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _get_model_dir_from_config(vllm_config: VllmConfig) -> Path:
    model = vllm_config.model_config.model
    return Path(model).expanduser().resolve()


@lru_cache(maxsize=8)
def _load_remote_audio_module(model_dir: str) -> ModuleType:
    return load_local_module(
        model_dir,
        "modeling_audio.py",
        "gpa15_local_audio",
    )


def _activation(name: str) -> nn.Module:
    return {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "selu": nn.SELU(),
    }.get(name, nn.GELU())


class AudioMLPAdapter(nn.Module):
    def __init__(self, config: ArkasrConfig, model_dir: Path) -> None:
        super().__init__()
        audio_module = _load_remote_audio_module(str(model_dir))
        whisper_encoder_cls = audio_module.WhisperSpecialEncoder

        whisper_config = config.whisper_config
        self.merge_factor = int(config.merge_factor)
        self.whisper = whisper_encoder_cls(
            whisper_config,
            use_rope=getattr(config, "use_rope", True),
        )
        self.whisper.layer_norm = nn.Identity()
        self.layer_norm = nn.LayerNorm(whisper_config.hidden_size)

        input_dim = whisper_config.hidden_size * self.merge_factor
        output_dim = config.hidden_size
        self.adapting = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            _activation(getattr(config, "mlp_adapter_act", "gelu")),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, audios: torch.Tensor) -> torch.Tensor:
        batch_size = audios.size(0)
        encoded = self.whisper(audios)[0]
        encoded = self.layer_norm(encoded)

        seq_len = encoded.size(1)
        if seq_len % self.merge_factor != 0:
            target_len = (seq_len // self.merge_factor) * self.merge_factor
            if target_len <= 0:
                target_len = self.merge_factor
                pad_len = target_len - seq_len
                if pad_len > 0:
                    pad = encoded.new_zeros((batch_size, pad_len, encoded.size(-1)))
                    encoded = torch.cat([encoded, pad], dim=1)
            else:
                encoded = encoded[:, :target_len, :]

        encoded = encoded.reshape(batch_size, -1, encoded.size(-1) * self.merge_factor)
        return self.adapting(encoded)


def _arkasr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return {
        "audios": MultiModalFieldConfig.batched("audio"),
    }


class ArkasrMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audios"},
                fields_factory=_arkasr_field_config,
            )
        return super()._parse_audio_data(data)


class ArkasrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> ArkasrConfig:
        return self.ctx.get_hf_config(ArkasrConfig)

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        if kwargs:
            # GPA's local processor accepts runtime kwargs in apply_chat_template
            # rather than at construction time.
            pass
        return load_local_processor(str(Path(self.ctx.model_config.model)))

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        processor = self.get_hf_processor(**kwargs)
        feature_extractor = processor.feature_extractor
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del seq_len, mm_counts
        processor = self.get_hf_processor()
        feature_extractor = processor.feature_extractor
        max_audio_samples = int(
            getattr(feature_extractor, "chunk_length", 30)
            * getattr(feature_extractor, "sampling_rate", 16000)
        )
        hop_length = int(getattr(feature_extractor, "hop_length", 160))
        mel_frames = max_audio_samples // max(hop_length, 1)
        return {
            "audio": ArkasrMultiModalProcessor._audio_token_count_from_mel_frames(
                mel_frames,
                self.get_hf_config(),
            )
        }

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return ArkasrMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
        )


class ArkasrDummyInputsBuilder(BaseDummyInputsBuilder[ArkasrProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        processor = self.info.get_hf_processor()
        return getattr(processor, "audio_token", "<|audio|>") * mm_counts.get(
            "audio", 0
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        del seq_len

        feature_extractor = self.info.get_feature_extractor()
        audio_overrides = mm_options.get("audio") if mm_options else None
        length = (
            audio_overrides.length
            if audio_overrides and audio_overrides.length
            else int(1 * feature_extractor.sampling_rate)
        )
        return {
            "audio": self._get_dummy_audios(
                length=length,
                num_audios=mm_counts.get("audio", 0),
                overrides=audio_overrides,
            )
        }


class ArkasrMultiModalProcessor(BaseMultiModalProcessor[ArkasrProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processor = self.info.get_hf_processor(**mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        audio = mm_data.get("audios", mm_data.get("audio"))

        if audio is None:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return BatchFeature({"input_ids": [token_ids]}, tensor_type="pt")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "array": audio},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        kwargs = {
            "return_tensors": "pt",
            "sampling_rate": processor.feature_extractor.sampling_rate,
            "audio_padding": "longest",
            "add_generation_prompt": True,
            "text_kwargs": {"padding": "longest"},
            **mm_kwargs,
        }
        kwargs.update(tok_kwargs)
        return processor.apply_chat_template(conversation, **kwargs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _arkasr_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        config = self.info.get_hf_config()
        audio_token = getattr(processor, "audio_token", "<|audio|>")
        audio_token_id = getattr(config, "audio_token_id", None)
        if audio_token_id is None:
            audio_token_id = tokenizer.get_vocab()[audio_token]

        out_mm_data = out_mm_kwargs.get_data()
        audios = out_mm_data.get("audios")
        if audios is None:
            audio_token_counts: list[int] = []
        else:
            if isinstance(audios, torch.Tensor) and audios.ndim == 2:
                audios = audios.unsqueeze(0)
            audio_token_counts = [
                self._audio_token_count_from_features(
                    audios[item_idx], processor, config
                )
                for item_idx in range(len(audios))
            ]

        def get_replacement(item_idx: int):
            if audio_token_counts:
                num_features = audio_token_counts[item_idx]
            else:
                audio_items = mm_items.get_items("audio", AudioProcessorItems)
                audio = audio_items.get(item_idx)
                raw_len = len(audio)
                num_features = self._audio_token_count_from_raw_len(
                    raw_len, processor, config
                )
            return [int(audio_token_id)] * int(num_features)

        return [
            PromptReplacement(
                modality="audio",
                target=[int(audio_token_id)],
                replacement=get_replacement,
            )
        ]

    @staticmethod
    def _audio_token_count_from_features(
        audio_features: torch.Tensor,
        processor: ProcessorMixin,
        config: ArkasrConfig,
    ) -> int:
        del processor
        frames = int(audio_features.shape[-1])
        return ArkasrMultiModalProcessor._audio_token_count_from_mel_frames(
            frames,
            config,
        )

    @staticmethod
    def _audio_token_count_from_mel_frames(
        mel_frames: int,
        config: ArkasrConfig,
    ) -> int:
        downsampled = (int(mel_frames) + 1) // 2
        merged = downsampled // max(int(config.merge_factor), 1)
        return max(int(merged), 1)

    @staticmethod
    def _audio_token_count_from_raw_len(
        raw_len: int,
        processor: ProcessorMixin,
        config: ArkasrConfig,
    ) -> int:
        hop_length = int(getattr(processor.feature_extractor, "hop_length", 160))
        mel_frames = int(raw_len) // max(hop_length, 1)
        downsampled = (mel_frames + 1) // 2
        merged = downsampled // max(int(config.merge_factor), 1)
        return max(int(merged), 1)


@MULTIMODAL_REGISTRY.register_processor(
    ArkasrMultiModalProcessor,
    info=ArkasrProcessingInfo,
    dummy_inputs=ArkasrDummyInputsBuilder,
)
class ArkasrForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio|>"
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        if not isinstance(config, ArkasrConfig):
            config = ArkasrConfig.from_dict(config.to_dict())

        self.config = config
        self.audio_token_id = int(config.audio_token_id)
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = AudioMLPAdapter(
                config,
                model_dir=_get_model_dir_from_config(vllm_config),
            )

        text_config = self._to_qwen2_config(config)
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen2ForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    text_config,
                    architectures=["Qwen2ForCausalLM"],
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @staticmethod
    def _to_qwen2_config(config: ArkasrConfig) -> Qwen2Config:
        text_dict = copy.deepcopy(config.to_dict())
        text_dict.pop("whisper_config", None)
        text_dict["model_type"] = "qwen2"
        text_dict["architectures"] = ["Qwen2ForCausalLM"]
        return Qwen2Config.from_dict(text_dict)

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> torch.Tensor | None:
        audios = kwargs.pop("audios", None)
        if audios is None:
            return None
        if isinstance(audios, torch.Tensor) and audios.ndim == 2:
            audios = audios.unsqueeze(0)
        if isinstance(audios, list):
            if not audios:
                return None
            if not all(isinstance(audio, torch.Tensor) for audio in audios):
                raise ValueError("Expected all audio items to be tensors")
            audios = torch.nn.utils.rnn.pad_sequence(
                [audio.transpose(0, 1) for audio in audios],
                batch_first=True,
            ).transpose(1, 2)
        if not isinstance(audios, torch.Tensor):
            raise ValueError(f"Expected audios to be a tensor, got {type(audios)}")
        return audios

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict[str, Any]:
        if "audios" not in kwargs:
            return {}
        audios = self._parse_and_validate_audio_input(**kwargs)
        return {"audio": audios} if audios is not None else {}

    def _process_audio_input(self, audios: torch.Tensor) -> tuple[torch.Tensor, ...]:
        target_dtype = next(self.audio_encoder.parameters()).dtype
        audio_embeddings = self.audio_encoder(audios.to(dtype=target_dtype))
        return tuple(audio_embeddings.unbind(dim=0))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []
        return self._process_audio_input(mm_input_by_modality["audio"])

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        embed_kwargs: dict[str, object] = {"is_multimodal": is_multimodal}
        if "handle_oov_mm_token" in inspect.signature(
            self._embed_text_input_ids
        ).parameters:
            embed_kwargs["handle_oov_mm_token"] = handle_oov_mm_token
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            **embed_kwargs,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            is_multimodal = input_ids == self.audio_token_id
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["language_model.lm_head."]
                if self.config.tie_word_embeddings
                else None
            ),
            ignore_unexpected_suffixes=[".num_batches_tracked"],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            tower_model="audio_encoder.",
        )

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: str,
    ) -> SpeechToTextConfig:
        processor = load_local_processor(model_config.model)
        feature_extractor: WhisperFeatureExtractor = processor.feature_extractor
        return SpeechToTextConfig(
            max_audio_clip_s=feature_extractor.chunk_length,
            sample_rate=feature_extractor.sampling_rate,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        del stt_config, language, to_language
        if task_type not in ("transcribe", "translate"):
            raise ValueError(f"Unsupported task_type: {task_type}")
        tokenizer = cached_tokenizer_from_config(
            model_config,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        prompt = request_prompt or "Please transcribe this audio."
        prompt_ids = tokenizer.encode(
            f"<|user|><|begin_of_audio|><|audio|><|end_of_audio|>{prompt}"
            "<|assistant|>",
            add_special_tokens=False,
        )
        return TokensPrompt(
            prompt_token_ids=prompt_ids,
            multi_modal_data={"audio": audio},
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return text.strip()
