import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf
from tts_han_char_tokenizer import create_tts_han_char_processor, encode_tts_content_text
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

DEFAULT_HF_HOME = str(Path(tempfile.gettempdir()) / "ark_audio_hf_home")
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(DEFAULT_HF_HOME) / "transformers"))
os.environ.setdefault("HF_MODULES_CACHE", str(Path(DEFAULT_HF_HOME) / "modules"))

from transformers import AutoProcessor, AutoTokenizer, GenerationConfig, PretrainedConfig


SEM_RE = re.compile(r"<\|bicodec_semantic_(\d+)\|>")
START_CONTENT_TAG = "<|start_content|>"
END_CONTENT_TAG = "<|end_content|>"
MAIN_MODEL_FILENAMES = {
    "fp32": "llm_kv_cpu_fp32.onnx",
    "fp16": "llm_kv_cuda_fp16.onnx",
    "int8": "llm_kv_cpu_fp32_int8.onnx",
    "int4": "llm_kv_cpu_fp32_int4.onnx",
}
AUDIO_ENCODER_FILENAMES = {
    "fp16": ("audio_encoder_whisper_fp16.onnx", "audio_encoder_adapter_fp16.onnx"),
    "int4": ("audio_encoder_whisper_int8.onnx", "audio_encoder_adapter_int8.onnx"),
}
MODEL_META_FILENAMES = {
    "fp32": "llm_kv_fp32_qwen_native.json",
    "fp16": "llm_kv_fp16_qwen_native.json",
    "int8": "llm_kv_fp32_qwen_native.json",
    "int4": "llm_kv_fp32_qwen_native.json",
}
LEGACY_MODEL_META_FILENAMES = {
    "fp32": "llm_kv_fp32.json",
    "fp16": "llm_kv_fp16_manual.json",
    "int8": "llm_kv_fp32.json",
    "int4": "llm_kv_fp32.json",
}
DEFAULT_ASR_BLOCK_TOKEN_ID_FROM = 151670


class BlockTokenIdsFromLogitsProcessor(LogitsProcessor):
    """Mask token ids >= block_from_id during generation."""

    def __init__(self, block_from_id: int):
        self.block_from_id = int(block_from_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        if self.block_from_id < vocab_size:
            scores[:, self.block_from_id:] = -float("inf")
        return scores

    def apply_numpy(self, scores: np.ndarray) -> np.ndarray:
        vocab_size = scores.shape[-1]
        if self.block_from_id < vocab_size:
            scores[self.block_from_id:] = -np.inf
        return scores


def load_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def ort_dtype_to_numpy(ort_type: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if ort_type not in mapping:
        raise ValueError(f"unsupported ORT tensor type: {ort_type}")
    return mapping[ort_type]


def as_dict(batch) -> Dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        return batch
    return {k: batch[k] for k in batch.keys()}


def normalize_global_tokens(global_tokens: np.ndarray) -> np.ndarray:
    global_tokens = np.asarray(global_tokens, dtype=np.int64)
    if global_tokens.ndim == 1:
        global_tokens = global_tokens[np.newaxis, np.newaxis, :]
    elif global_tokens.ndim == 2:
        global_tokens = global_tokens[:, np.newaxis, :]
    if tuple(global_tokens.shape) != (1, 1, 32):
        raise ValueError(f"unexpected global token shape: {global_tokens.shape}")
    return np.ascontiguousarray(global_tokens)


def build_tts_inputs(tts_processor, text: str, global_tokens: np.ndarray) -> Dict[str, torch.Tensor]:
    tokenizer = tts_processor.tokenizer
    global_text = "".join(f"<|bicodec_global_{int(x)}|>" for x in global_tokens.reshape(-1).tolist())
    prompt = (
        "Given the reference audio, synthesize speech for the following text in the same voice."
        f"<|start_global_token|>{global_text}<|end_global_token|>"
        f"{START_CONTENT_TAG}{text.strip()}{END_CONTENT_TAG}"
    )
    template_text = tts_processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    start_idx = template_text.index(START_CONTENT_TAG) + len(START_CONTENT_TAG)
    end_idx = template_text.index(END_CONTENT_TAG, start_idx)
    prefix_text = template_text[:start_idx]
    content_text = template_text[start_idx:end_idx]
    suffix_text = template_text[end_idx:]

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    content_ids = encode_tts_content_text(tokenizer, content_text)["input_ids"]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prefix_ids + content_ids + suffix_ids], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
    }


def load_manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / "runtime_manifest.json").read_text())


def create_empty_cache(num_layers: int, max_total_len: int, num_kv_heads: int, head_dim: int, dtype: np.dtype) -> List[np.ndarray]:
    caches = []
    for _ in range(num_layers):
        caches.append(np.zeros((1, max_total_len, num_kv_heads, head_dim), dtype=dtype))
        caches.append(np.zeros((1, max_total_len, num_kv_heads, head_dim), dtype=dtype))
    return caches


def build_attention_mask(max_total_len: int, valid_len: int) -> np.ndarray:
    mask = np.zeros((1, max_total_len), dtype=np.int64)
    mask[:, :valid_len] = 1
    return mask


class OnnxPastKeyValues:
    def __init__(self, caches: List[np.ndarray], total_len: int):
        self.caches = caches
        self.total_len = int(total_len)


class OnnxGenerationWrapper(torch.nn.Module, GenerationMixin):
    main_input_name = "input_ids"
    _is_stateful = False

    def __init__(self, runtime: "ArkAudioOnnxRuntime", main_model_precision: Optional[str], device: torch.device):
        super().__init__()
        self.runtime = runtime
        self.main_model_precision = main_model_precision
        self.config = PretrainedConfig()
        self.config.is_encoder_decoder = False
        self.config.vocab_size = len(runtime.tokenizer)
        self.config.eos_token_id = 151665
        self.config.pad_token_id = int(runtime.manifest["pad_token_id"])
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.generation_config.eos_token_id = 151665
        self.generation_config.pad_token_id = int(runtime.manifest["pad_token_id"])
        self.register_buffer("_device_indicator", torch.empty(0, device=device), persistent=False)

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def _supports_default_dynamic_cache(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        return self._device_indicator.device

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value) -> None:
        raise NotImplementedError("ONNX runtime wrapper does not expose torch embeddings")

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[OnnxPastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Dict[str, object]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            start = int(past_key_values.total_len)
        else:
            start = 0
        if cache_position is None or cache_position.shape[0] != input_ids.shape[1]:
            cache_position = torch.arange(
                start,
                start + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[OnnxPastKeyValues] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        llm_session, llm_embed_dtype, llm_cache_dtype, llm_filename = self.runtime._get_llm_session(self.main_model_precision)
        llm_meta = self.runtime._get_llm_meta(llm_filename)
        hidden_size = int(llm_meta["hidden_size"])
        num_layers = int(llm_meta["num_layers"])
        num_kv_heads = int(llm_meta["num_key_value_heads"])
        head_dim = hidden_size // int(llm_meta["num_attention_heads"])
        max_total_len = int(llm_meta["max_total_len"])

        if past_key_values is None:
            caches = create_empty_cache(num_layers, max_total_len, num_kv_heads, head_dim, llm_cache_dtype)
            total_len = 0
        else:
            caches = past_key_values.caches
            total_len = int(past_key_values.total_len)

        input_ids_np = input_ids.detach().to("cpu").numpy().astype(np.int64)
        inputs_embeds = self.runtime._embed(input_ids_np).astype(llm_embed_dtype)
        seq_len = input_ids_np.shape[1]
        if cache_position is None:
            cache_position_np = np.arange(total_len, total_len + seq_len, dtype=np.int64)
        else:
            cache_position_np = cache_position.detach().to("cpu").numpy().astype(np.int64)

        feeds = {
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position_np,
        }
        llm_input_names = {inp.name for inp in llm_session.get_inputs()}
        if "attention_mask" in llm_input_names:
            valid_len = total_len + seq_len if attention_mask is None else int(attention_mask.shape[1])
            feeds["attention_mask"] = build_attention_mask(max_total_len, valid_len)
        for i in range(num_layers):
            feeds[f"cache_key_{i}"] = caches[2 * i]
            feeds[f"cache_value_{i}"] = caches[2 * i + 1]

        outputs = llm_session.run(None, feeds)
        logits = outputs[0]
        deltas = outputs[1:]
        for i in range(num_layers):
            caches[2 * i][:, total_len : total_len + seq_len] = deltas[2 * i]
            caches[2 * i + 1][:, total_len : total_len + seq_len] = deltas[2 * i + 1]
        next_past = OnnxPastKeyValues(caches=caches, total_len=total_len + seq_len)
        logits_t = torch.from_numpy(logits).to(input_ids.device)
        return CausalLMOutputWithPast(logits=logits_t, past_key_values=next_past)


class ArkAudioOnnxRuntime:
    def __init__(self, runtime_root: Path):
        self.runtime_root = runtime_root.resolve()
        self.model_dir = self.runtime_root / "model"
        self.build_dir = self.runtime_root / "build"
        self.manifest = load_manifest(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.tts_processor = create_tts_han_char_processor(str(self.model_dir), trust_remote_code=True)
        self.tts_text_tokenizer_mode_name = "han_char"
        self.embedding = load_session(self.model_dir / self.manifest["embedding_default"])
        self.llm_sessions: Dict[str, ort.InferenceSession] = {}
        self.llm_input_dtypes: Dict[str, Tuple[np.dtype, np.dtype]] = {}
        self.audio_encoder_sessions: Dict[str, ort.InferenceSession] = {}
        self.audio_adapter_sessions: Dict[str, ort.InferenceSession] = {}
        self.detok_int8 = None
        self.detok_fp16 = None
        self.llm_meta_by_filename: Dict[str, dict] = {}
        self.generation_wrappers: Dict[str, OnnxGenerationWrapper] = {}
        self.embedding_input_dtype = ort_dtype_to_numpy(self.embedding.get_inputs()[0].type)
        self.audio_input_dtypes: Dict[str, np.dtype] = {}
        self.audio_adapter_input_dtypes: Dict[str, np.dtype] = {}
        self.default_llm_filename = self.manifest["llm_default"]
        self._get_llm_session(None)

        self.semantic_token_id_map = {}
        added_tokens_path = self.model_dir / "added_tokens.json"
        if added_tokens_path.exists():
            payload = json.loads(added_tokens_path.read_text())
            for token_text, token_id in payload.items():
                m = SEM_RE.fullmatch(token_text)
                if m:
                    self.semantic_token_id_map[int(token_id)] = int(m.group(1))

    def available_main_model_precisions(self) -> List[str]:
        available = []
        for precision, filename in MAIN_MODEL_FILENAMES.items():
            if (self.model_dir / filename).exists():
                available.append(precision)
        return available

    def default_main_model_precision(self) -> str:
        for precision, filename in MAIN_MODEL_FILENAMES.items():
            if filename == self.default_llm_filename:
                return precision
        return "custom"

    def tts_text_tokenizer_mode(self) -> str:
        return self.tts_text_tokenizer_mode_name

    def resolve_main_model_filename(self, main_model_precision: Optional[str]) -> str:
        if main_model_precision is None:
            return self.default_llm_filename
        if main_model_precision not in MAIN_MODEL_FILENAMES:
            raise ValueError(f"unsupported main model precision: {main_model_precision}")
        filename = MAIN_MODEL_FILENAMES[main_model_precision]
        model_path = self.model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"main model not found for precision '{main_model_precision}': {model_path}")
        return filename

    def _get_llm_precision_by_filename(self, filename: str) -> Optional[str]:
        for precision, mapped_filename in MAIN_MODEL_FILENAMES.items():
            if mapped_filename == filename:
                return precision
        return None

    def _get_llm_meta(self, filename: str) -> dict:
        meta = self.llm_meta_by_filename.get(filename)
        if meta is not None:
            return meta

        precision = self._get_llm_precision_by_filename(filename)
        candidates: List[Path] = []
        if precision is not None:
            candidates.append(self.build_dir / MODEL_META_FILENAMES[precision])
            candidates.append(self.build_dir / LEGACY_MODEL_META_FILENAMES[precision])
        candidates.append(self.build_dir / f"{Path(filename).stem}.json")

        for path in candidates:
            if path.exists():
                meta = json.loads(path.read_text())
                self.llm_meta_by_filename[filename] = meta
                return meta
        raise FileNotFoundError(f"llm meta json not found for model file: {filename}")

    def _get_llm_session(self, main_model_precision: Optional[str]) -> Tuple[ort.InferenceSession, np.dtype, np.dtype, str]:
        filename = self.resolve_main_model_filename(main_model_precision)
        session = self.llm_sessions.get(filename)
        dtypes = self.llm_input_dtypes.get(filename)
        if session is None or dtypes is None:
            session = load_session(self.model_dir / filename)
            embed_dtype = ort_dtype_to_numpy(session.get_inputs()[0].type)
            cache_input = next(inp for inp in session.get_inputs() if inp.name.startswith("cache_key_0"))
            cache_dtype = ort_dtype_to_numpy(cache_input.type)
            self.llm_sessions[filename] = session
            self.llm_input_dtypes[filename] = (embed_dtype, cache_dtype)
            dtypes = (embed_dtype, cache_dtype)
        return session, dtypes[0], dtypes[1], filename

    def _get_detokenizer(self, precision: str) -> ort.InferenceSession:
        if precision == "fp16":
            if self.detok_fp16 is None:
                self.detok_fp16 = load_session(self.model_dir / "spark_detokenizer_fp16.onnx")
            return self.detok_fp16
        if self.detok_int8 is None:
            self.detok_int8 = load_session(self.model_dir / "spark_detokenizer_int8.onnx")
        return self.detok_int8

    def resolve_audio_encoder_filenames(self, main_model_precision: Optional[str]) -> Tuple[str, str]:
        precision = main_model_precision if main_model_precision is not None else self.default_main_model_precision()
        encoder_filename, adapter_filename = AUDIO_ENCODER_FILENAMES.get(
            precision,
            (self.manifest["audio_encoder_default"], self.manifest["audio_adapter_default"]),
        )
        for filename in (encoder_filename, adapter_filename):
            if not (self.model_dir / filename).exists():
                raise FileNotFoundError(f"audio encoder asset not found for precision '{precision}': {filename}")
        return encoder_filename, adapter_filename

    def _ensure_audio_sessions(
        self,
        main_model_precision: Optional[str],
    ) -> Tuple[ort.InferenceSession, ort.InferenceSession, np.dtype, np.dtype]:
        encoder_filename, adapter_filename = self.resolve_audio_encoder_filenames(main_model_precision)
        if encoder_filename not in self.audio_encoder_sessions:
            encoder = load_session(self.model_dir / encoder_filename)
            self.audio_encoder_sessions[encoder_filename] = encoder
            self.audio_input_dtypes[encoder_filename] = ort_dtype_to_numpy(encoder.get_inputs()[0].type)
        if adapter_filename not in self.audio_adapter_sessions:
            adapter = load_session(self.model_dir / adapter_filename)
            self.audio_adapter_sessions[adapter_filename] = adapter
            self.audio_adapter_input_dtypes[adapter_filename] = ort_dtype_to_numpy(adapter.get_inputs()[0].type)
        return (
            self.audio_encoder_sessions[encoder_filename],
            self.audio_adapter_sessions[adapter_filename],
            self.audio_input_dtypes[encoder_filename],
            self.audio_adapter_input_dtypes[adapter_filename],
        )

    def _get_generation_wrapper(self, main_model_precision: Optional[str]) -> OnnxGenerationWrapper:
        filename = self.resolve_main_model_filename(main_model_precision)
        wrapper = self.generation_wrappers.get(filename)
        if wrapper is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            wrapper = OnnxGenerationWrapper(self, self._get_llm_precision_by_filename(filename), device)
            self.generation_wrappers[filename] = wrapper
        return wrapper

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        return self.embedding.run(None, {"input_ids": input_ids.astype(np.int64)})[0]

    def _encode_audio(self, audios: np.ndarray, main_model_precision: Optional[str]) -> np.ndarray:
        audio_encoder, audio_adapter, audio_input_dtype, audio_adapter_input_dtype = self._ensure_audio_sessions(
            main_model_precision
        )
        encoded = audio_encoder.run(None, {"audios": audios.astype(audio_input_dtype)})[0]
        merged = self._merge_audio_features(encoded)
        return audio_adapter.run(None, {"merged_audio_features": merged.astype(audio_adapter_input_dtype)})[0]

    def _merge_audio_features(self, encoded: np.ndarray) -> np.ndarray:
        merge_factor = int(self.manifest["audio_merge_factor"])
        hidden = int(self.manifest["audio_encoder_hidden_size"])
        bsz, seq_len, feat_dim = encoded.shape
        if feat_dim != hidden:
            raise ValueError(f"unexpected audio encoder hidden size: {feat_dim} != {hidden}")

        if seq_len < merge_factor:
            padded = np.zeros((bsz, merge_factor, hidden), dtype=encoded.dtype)
            padded[:, :seq_len, :] = encoded
            encoded_use = padded
        else:
            target_len = (seq_len // merge_factor) * merge_factor
            encoded_use = encoded[:, :target_len, :]

        new_seq = encoded_use.shape[1] // merge_factor
        return np.ascontiguousarray(encoded_use.reshape(bsz, new_seq, hidden * merge_factor))

    def _inject_audio(
        self,
        input_ids: np.ndarray,
        inputs_embeds: np.ndarray,
        audios: np.ndarray,
        main_model_precision: Optional[str],
    ) -> np.ndarray:
        audio_token_id = int(self.manifest["audio_token_id"])
        mask = input_ids == audio_token_id
        per_counts = mask.sum(axis=1)
        need_idx = np.nonzero(per_counts > 0)[0]
        if need_idx.size == 0:
            return inputs_embeds
        feats_sub = self._encode_audio(audios[need_idx], main_model_precision)
        hidden = inputs_embeds.shape[-1]
        for k, i in enumerate(need_idx.tolist()):
            n_i = int(per_counts[i])
            feat_i = feats_sub[k]
            sa = feat_i.shape[0]
            if sa < n_i:
                feat_i = np.concatenate([feat_i, np.zeros((n_i - sa, hidden), dtype=feat_i.dtype)], axis=0)
            elif sa > n_i:
                feat_i = feat_i[:n_i]
            pos_i = np.nonzero(mask[i])[0]
            inputs_embeds[i, pos_i, :] = feat_i
        return inputs_embeds

    def _forward_prefill(
        self,
        llm_session: ort.InferenceSession,
        llm_embed_dtype: np.dtype,
        llm_cache_dtype: np.dtype,
        llm_meta: dict,
        inputs_embeds: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray], int]:
        hidden_size = int(llm_meta["hidden_size"])
        num_layers = int(llm_meta["num_layers"])
        num_kv_heads = int(llm_meta["num_key_value_heads"])
        head_dim = hidden_size // int(llm_meta["num_attention_heads"])
        max_total_len = int(llm_meta["max_total_len"])

        caches = create_empty_cache(num_layers, max_total_len, num_kv_heads, head_dim, llm_cache_dtype)
        prompt_len = inputs_embeds.shape[1]
        feeds = {
            "inputs_embeds": inputs_embeds.astype(llm_embed_dtype),
            "cache_position": np.arange(prompt_len, dtype=np.int64),
        }
        llm_input_names = {inp.name for inp in llm_session.get_inputs()}
        if "attention_mask" in llm_input_names:
            feeds["attention_mask"] = build_attention_mask(max_total_len, prompt_len)
        for i in range(num_layers):
            feeds[f"cache_key_{i}"] = caches[2 * i]
            feeds[f"cache_value_{i}"] = caches[2 * i + 1]
        outputs = llm_session.run(None, feeds)
        logits = outputs[0]
        deltas = outputs[1:]
        for i in range(num_layers):
            k_delta = deltas[2 * i]
            v_delta = deltas[2 * i + 1]
            caches[2 * i][:, :prompt_len] = k_delta
            caches[2 * i + 1][:, :prompt_len] = v_delta
        return logits, caches, prompt_len

    def _decode_loop(
        self,
        prompt_input_ids: np.ndarray,
        prompt_embeds: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        do_sample: bool,
        main_model_precision: Optional[str] = None,
        stop_ids: Optional[set[int]] = None,
        asr_block_token_id_from: Optional[int] = None,
    ) -> List[int]:
        llm_session, llm_embed_dtype, llm_cache_dtype, llm_filename = self._get_llm_session(main_model_precision)
        llm_meta = self._get_llm_meta(llm_filename)
        max_total_len = int(llm_meta["max_total_len"])
        logits, caches, total_len = self._forward_prefill(
            llm_session,
            llm_embed_dtype,
            llm_cache_dtype,
            llm_meta,
            prompt_embeds,
        )
        stop_ids = set(int(x) for x in (stop_ids if stop_ids is not None else self.manifest["stop_token_ids"]))
        token_ids: List[int] = []
        history_ids = prompt_input_ids.reshape(-1).tolist()
        logits_processor = None
        if asr_block_token_id_from is not None and int(asr_block_token_id_from) >= 0:
            logits_processor = BlockTokenIdsFromLogitsProcessor(int(asr_block_token_id_from))

        def select_next_token(step_logits: np.ndarray) -> int:
            adjusted = np.array(step_logits, copy=True)
            if logits_processor is not None:
                adjusted = logits_processor.apply_numpy(adjusted)
            if repetition_penalty > 0 and repetition_penalty != 1.0 and history_ids:
                repeated = np.asarray(sorted(set(history_ids)), dtype=np.int64)
                repeated_logits = adjusted[repeated]
                adjusted[repeated] = np.where(
                    repeated_logits < 0,
                    repeated_logits * repetition_penalty,
                    repeated_logits / repetition_penalty,
                )
            if do_sample and temperature > 0:
                probs = torch.softmax(torch.from_numpy(adjusted) / float(temperature), dim=-1)
                return int(torch.multinomial(probs, num_samples=1).item())
            return int(np.argmax(adjusted))

        step_logits = np.array(logits[0, 0], copy=True)
        next_token = select_next_token(step_logits)
        token_ids.append(next_token)
        history_ids.append(next_token)

        num_layers = int(llm_meta["num_layers"])
        for _ in range(max_new_tokens - 1):
            if token_ids[-1] in stop_ids:
                break
            embed = self._embed(np.array([[token_ids[-1]]], dtype=np.int64))
            feeds = {
                "inputs_embeds": embed.astype(llm_embed_dtype),
                "cache_position": np.array([total_len], dtype=np.int64),
            }
            llm_input_names = {inp.name for inp in llm_session.get_inputs()}
            if "attention_mask" in llm_input_names:
                feeds["attention_mask"] = build_attention_mask(max_total_len, total_len + 1)
            for i in range(num_layers):
                feeds[f"cache_key_{i}"] = caches[2 * i]
                feeds[f"cache_value_{i}"] = caches[2 * i + 1]
            outputs = llm_session.run(None, feeds)
            logits = outputs[0]
            deltas = outputs[1:]
            for i in range(num_layers):
                caches[2 * i][:, total_len:total_len + 1] = deltas[2 * i]
                caches[2 * i + 1][:, total_len:total_len + 1] = deltas[2 * i + 1]
            total_len += 1

            step_logits = np.array(logits[0, 0], copy=True)
            next_token = select_next_token(step_logits)
            token_ids.append(next_token)
            history_ids.append(next_token)
        return token_ids

    def run_asr(
        self,
        audio_path: str,
        begin_time: float,
        end_time: float,
        max_audio_seconds: int,
        max_new_tokens: int,
        asr_block_token_id_from: Optional[int] = DEFAULT_ASR_BLOCK_TOKEN_ID_FROM,
        main_model_precision: Optional[str] = None,
    ) -> str:
        conv = [{
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path, "begin_time": begin_time, "end_time": end_time},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        }]
        inputs_raw = self.processor.apply_chat_template(
            conv,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
            add_generation_prompt=True,
            text_kwargs={"padding": "longest"},
            audio_max_length=int(max_audio_seconds * 16000),
        )
        inputs = as_dict(inputs_raw)
        input_ids = inputs["input_ids"].cpu().numpy().astype(np.int64)
        audios = inputs["audios"].float().cpu().numpy().astype(np.float32)
        embeds = self._embed(input_ids)
        embeds = self._inject_audio(input_ids, embeds, audios, main_model_precision)
        token_ids = self._decode_loop(
            input_ids,
            embeds,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.0,
            do_sample=False,
            main_model_precision=main_model_precision,
            asr_block_token_id_from=asr_block_token_id_from,
        )
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def run_tts(
        self,
        text: str,
        output_path: Path,
        global_token_path: Optional[Path],
        max_new_tokens: int,
        decoder_precision: str,
        temperature: float = 0.1,
        repetition_penalty: float = 1.2,
        main_model_precision: Optional[str] = None,
    ) -> Tuple[List[int], str]:
        if global_token_path is None:
            global_token_path = self.model_dir / "reference" / "default_global_tokens.npy"
        global_tokens = normalize_global_tokens(np.load(global_token_path))
        inputs = build_tts_inputs(self.tts_processor, text, global_tokens)

        generation_model = self._get_generation_wrapper(main_model_precision)
        generation_device = generation_model._device_indicator.device
        input_ids_t = inputs["input_ids"].to(generation_device)
        attention_mask_t = inputs["attention_mask"].to(generation_device)
        prompt_len = input_ids_t.shape[1]
        gen = generation_model.generate(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )
        gen_ids = gen[0, prompt_len:].detach().to("cpu")
        if gen_ids.numel() > 0 and int(gen_ids[-1].item()) == 151665:
            gen_ids = gen_ids[:-1]
        token_ids = gen_ids.tolist()
        gen_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

        sem_ids = [self.semantic_token_id_map[t] for t in token_ids if t in self.semantic_token_id_map]
        if not sem_ids:
            sem_ids = [int(m.group(1)) for m in SEM_RE.finditer(gen_text)]
        if not sem_ids:
            raise RuntimeError("no semantic tokens produced")

        semantic_tokens = np.asarray(sem_ids, dtype=np.int64)[np.newaxis, :]
        detok = self._get_detokenizer(decoder_precision)
        audio = detok.run(None, {"semantic_tokens": semantic_tokens, "global_tokens": global_tokens.astype(np.int64)})[0]
        wav = np.asarray(audio).squeeze().astype(np.float32)
        sf.write(str(output_path), wav, int(self.manifest["sample_rate"]))
        return sem_ids, gen_text

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".wav",
        *,
        begin_time: float = -1,
        end_time: float = -1,
        max_audio_seconds: int = 30,
        max_new_tokens: int = 256,
        asr_block_token_id_from: Optional[int] = DEFAULT_ASR_BLOCK_TOKEN_ID_FROM,
        main_model_precision: Optional[str] = None,
    ) -> str:
        with tempfile.NamedTemporaryFile(prefix="ark_audio_asr_", suffix=suffix, delete=False) as handle:
            tmp_path = Path(handle.name)
            handle.write(audio_bytes)
        try:
            return self.run_asr(
                str(tmp_path),
                begin_time=begin_time,
                end_time=end_time,
                max_audio_seconds=max_audio_seconds,
                max_new_tokens=max_new_tokens,
                asr_block_token_id_from=asr_block_token_id_from,
                main_model_precision=main_model_precision,
            )
        finally:
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPA ONNX runtime")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--task", type=str, choices=["asr", "tts", "both"], required=True)
    parser.add_argument("--asr_audio", type=str)
    parser.add_argument("--begin_time", type=float, default=-1)
    parser.add_argument("--end_time", type=float, default=-1)
    parser.add_argument("--max_audio_seconds", type=int, default=30)
    parser.add_argument("--asr_max_new_tokens", type=int, default=256)
    parser.add_argument("--asr_block_token_id_from", type=int, default=DEFAULT_ASR_BLOCK_TOKEN_ID_FROM)
    parser.add_argument("--tts_text", type=str)
    parser.add_argument("--tts_out_wav", type=Path)
    parser.add_argument(
        "--voice-global-token",
        "--global-token-path",
        dest="voice_global_token",
        type=Path,
        help=(
            "Path to a voice global_tokens.npy file used for TTS voice control. "
            "If omitted, the CLI falls back to runtime-root/model/reference/default_global_tokens.npy."
        ),
    )
    parser.add_argument("--tts_max_new_tokens", type=int, default=1024)
    parser.add_argument("--decoder-precision", type=str, choices=["int8", "fp16"], default="fp16")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition-penalty", type=float, default=1.5)
    parser.add_argument("--main-model-precision", type=str, choices=["fp32", "fp16", "int8", "int4"])
    args = parser.parse_args()

    runtime = ArkAudioOnnxRuntime(args.runtime_root)
    if args.task in {"asr", "both"}:
        if not args.asr_audio:
            raise ValueError("--asr_audio is required for ASR")
        text = runtime.run_asr(
            args.asr_audio,
            args.begin_time,
            args.end_time,
            args.max_audio_seconds,
            args.asr_max_new_tokens,
            args.asr_block_token_id_from,
            main_model_precision=args.main_model_precision,
        )
        print("ASR_RESULT:", text)
    if args.task in {"tts", "both"}:
        if not args.tts_text or args.tts_out_wav is None:
            raise ValueError("--tts_text and --tts_out_wav are required for TTS")
        if args.voice_global_token is not None and not args.voice_global_token.exists():
            raise ValueError(f"voice global token file not found: {args.voice_global_token}")
        sem_ids, gen_text = runtime.run_tts(
            args.tts_text,
            args.tts_out_wav,
            args.voice_global_token,
            args.tts_max_new_tokens,
            args.decoder_precision,
            args.temperature,
            args.repetition_penalty,
            args.main_model_precision,
        )
        print("TTS_SEMANTIC_COUNT:", len(sem_ids))
        print("TTS_GEN_TEXT:", gen_text[:500])


if __name__ == "__main__":
    main()
