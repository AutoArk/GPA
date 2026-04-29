from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from tts_han_char_tokenizer import encode_tts_content_text


START_CONTENT_TAG = "<|start_content|>"
END_CONTENT_TAG = "<|end_content|>"
DEFAULT_FALLBACK_AUDIO = Path(__file__).resolve().parents[1] / "samples" / "sample.mp3"
DEFAULT_FALLBACK_TEXT = "This is a repository fallback transcription sample."


def rank0_print(*args: object) -> None:
    if _dist_rank() == 0:
        print(*args)


def _dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank() -> int:
    if _dist_available_and_initialized():
        return torch.distributed.get_rank()
    return 0


def _dist_barrier() -> None:
    if _dist_available_and_initialized():
        torch.distributed.barrier()


def load_json_dataset_rank0_build_all_load(data_file: str, cache_dir: str):
    from datasets import load_dataset

    rank = _dist_rank()
    if rank == 0:
        rank0_print(f"[datasets] rank0 building cache for: {data_file}")
        load_dataset("json", data_files=data_file, split="train", cache_dir=cache_dir)
        rank0_print(f"[datasets] rank0 build done for: {data_file}")

    _dist_barrier()
    return load_dataset("json", data_files=data_file, split="train", cache_dir=cache_dir)


def resolve_dataset_media_paths(dataset, data_file: str):
    base_dir = Path(data_file).expanduser().resolve().parent
    path_fields = ("audio", "pair_audio", "ref_audio", "vc_audio")

    def _resolve_feature_paths(feature: dict[str, Any]) -> dict[str, Any]:
        updated = dict(feature)
        for field_name in path_fields:
            field_value = updated.get(field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                continue
            candidate = Path(field_value)
            if candidate.is_absolute():
                updated[field_name] = str(candidate)
            else:
                updated[field_name] = str((base_dir / candidate).resolve())
        return updated

    return dataset.map(_resolve_feature_paths)


@dataclass
class ArkasrDataCollator:
    processor: Any
    tts_processor: Optional[Any] = None
    ignore_index: int = -100
    max_length: int = 1024
    max_audio_seconds: int = 30
    sampling_rate: int = 16000
    spark_tokenizer: Optional[Any] = None
    tts_missing_token_policy: str = "error"

    def _get_task(self, feature: dict[str, Any]) -> str:
        task_name = str(feature.get("task", "asr")).lower().strip()
        tts_aliases = {"tts", "tts-a", "tts_a", "tts-b", "tts_b"}
        return "tts" if task_name in tts_aliases else "asr"

    def _build_asr_conv(self, feature: dict[str, Any]) -> list[dict[str, Any]]:
        begin_time = feature.get("begin_time", -1)
        end_time = feature.get("end_time", -1)
        audio_path = feature.get("audio")
        text = feature.get("text")

        if not audio_path or not isinstance(audio_path, str) or not Path(audio_path).exists():
            if DEFAULT_FALLBACK_AUDIO.exists():
                rank0_print(f"[warn] audio path is invalid: {audio_path}. Falling back to {DEFAULT_FALLBACK_AUDIO}.")
                audio_path = str(DEFAULT_FALLBACK_AUDIO)
                text = DEFAULT_FALLBACK_TEXT
            else:
                raise FileNotFoundError(f"ASR sample audio path is invalid and no fallback sample exists: {audio_path}")

        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path, "begin_time": begin_time, "end_time": end_time},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": str(text).strip()}]},
        ]

    def _global_ids_to_str(self, global_ids: list[int]) -> str:
        return (
            "<|start_global_token|>"
            + "".join([f"<|bicodec_global_{int(token_id)}|>" for token_id in global_ids])
            + "<|end_global_token|>"
        )

    def _semantic_ids_to_str(self, semantic_ids: list[int]) -> str:
        return "".join([f"<|bicodec_semantic_{int(token_id)}|>" for token_id in semantic_ids])

    def _build_tts_convs_from_features(self, tts_features: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        conversations: list[list[dict[str, Any]]] = []

        for idx, feature in enumerate(tts_features):
            text = str(feature.get("text", "")).strip()
            global_ids = feature.get("pair_global_ids")
            semantic_ids = feature.get("audio_semantic_ids")

            if global_ids is None or semantic_ids is None:
                if self.tts_missing_token_policy != "fallback_to_tokenizer":
                    raise ValueError(
                        f"TTS sample idx={idx} is missing pair_global_ids/audio_semantic_ids. "
                        "Fix the JSONL record or set tts_missing_token_policy='fallback_to_tokenizer'."
                    )

                if self.spark_tokenizer is None:
                    raise ValueError(
                        f"TTS sample idx={idx} is missing pair_global_ids/audio_semantic_ids and "
                        "spark_tokenizer=None, so tokenizer fallback is unavailable."
                    )

                pair_audio = feature.get("pair_audio") or feature.get("ref_audio") or feature.get("audio")
                audio_path = feature.get("audio") or feature.get("pair_audio") or feature.get("ref_audio")
                if not pair_audio or not isinstance(pair_audio, str) or not Path(pair_audio).exists():
                    pair_audio = audio_path
                if not audio_path or not isinstance(audio_path, str) or not Path(audio_path).exists():
                    audio_path = pair_audio

                with torch.no_grad():
                    pair_output = self.spark_tokenizer.tokenize([pair_audio])
                    audio_output = self.spark_tokenizer.tokenize([audio_path])

                global_ids = pair_output["global_tokens"][0, 0].detach().cpu().tolist()
                semantic_ids = audio_output["semantic_tokens"][0].detach().cpu().tolist()

            if not isinstance(global_ids, (list, tuple)) or len(global_ids) != 32:
                raise ValueError(
                    f"TTS sample idx={idx}: pair_global_ids must be a list of length 32, got {type(global_ids)}."
                )
            if not isinstance(semantic_ids, (list, tuple)):
                raise ValueError(
                    f"TTS sample idx={idx}: audio_semantic_ids must be a list[int], got {type(semantic_ids)}."
                )

            trimmed_semantic_ids = list(semantic_ids)[: self.max_length]
            user_prompt = (
                "Given the reference audio, synthesize speech for the following text in the same voice."
                + self._global_ids_to_str(list(map(int, global_ids)))
                + START_CONTENT_TAG
                + text
                + END_CONTENT_TAG
            )
            assistant_output = self._semantic_ids_to_str(list(map(int, trimmed_semantic_ids)))
            conversations.append(
                [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_output},
                ]
            )

        return conversations

    def _make_labels_by_role_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        processor: Optional[Any] = None,
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, self.ignore_index)
        active_processor = processor or self.processor
        tokenizer = active_processor.tokenizer
        user_token_id = tokenizer.convert_tokens_to_ids(active_processor.user_token)
        assistant_token_id = tokenizer.convert_tokens_to_ids(active_processor.assistant_token)

        for row_index in range(input_ids.size(0)):
            is_training_turn = False
            for col_index in range(input_ids.size(1)):
                if attention_mask[row_index, col_index].item() == 0:
                    continue

                token_id = input_ids[row_index, col_index].item()
                if token_id == user_token_id:
                    if is_training_turn:
                        labels[row_index, col_index] = token_id
                        is_training_turn = False
                    else:
                        is_training_turn = False
                    continue

                if token_id == assistant_token_id:
                    is_training_turn = True
                    continue

                if is_training_turn:
                    labels[row_index, col_index] = token_id

        return labels

    def _build_asr_sub_batch(self, asr_convs: list[list[dict[str, Any]]], asr_processor: Any) -> dict[str, torch.Tensor]:
        tokenizer = asr_processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0

        templates_and_audios = asr_processor._build_templates_and_audios(
            conversations=asr_convs,
            sampling_rate=self.sampling_rate,
            add_generation_prompt=True,
        )
        if len(templates_and_audios) == 2:
            prompt_templates, audios_raw = templates_and_audios
            prompt_audio_counts = [prompt.count(asr_processor.bos_audio_token) for prompt in prompt_templates]
        elif len(templates_and_audios) == 3:
            prompt_templates, audios_raw, prompt_audio_counts = templates_and_audios
        else:
            raise ValueError(
                "processor._build_templates_and_audios must return (prompt_templates, audios_raw) "
                "or (prompt_templates, audios_raw, prompt_audio_counts)."
            )

        audio_max_length = int(self.max_audio_seconds * self.sampling_rate)
        features = asr_processor.feature_extractor(
            audios_raw,
            sampling_rate=self.sampling_rate,
            return_tensors="np",
            return_attention_mask=False,
            padding="longest",
            max_length=audio_max_length,
        )
        input_features = features["input_features"]
        audios = input_features if isinstance(input_features, torch.Tensor) else torch.tensor(input_features, dtype=torch.float32)
        audios = audios.to(dtype=torch.float32)

        if hasattr(asr_processor, "_calculate_audio_token_counts_per_sample"):
            audio_token_counts = asr_processor._calculate_audio_token_counts_per_sample(
                audios_raw=audios_raw,
                sampling_rate=self.sampling_rate,
                audio_max_length=audio_max_length,
                audio_pad_to_multiple_of=None,
            )
        else:
            hop_length = int(getattr(asr_processor.feature_extractor, "hop_length", 160))
            audio_token_counts = []
            for audio_raw in audios_raw:
                audio_frames = min(len(audio_raw), audio_max_length) // max(hop_length, 1)
                audio_token_counts.append(asr_processor.calculate_audio_token_count(int(audio_frames)))

        input_rows: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []
        audio_index = 0

        for template_text, audio_count in zip(prompt_templates, prompt_audio_counts):
            prompt_text = template_text
            for _ in range(audio_count):
                if audio_index >= len(audio_token_counts):
                    raise ValueError("Audio token count mismatch while building ASR prompts.")

                audio_start = prompt_text.index(asr_processor.bos_audio_token) + len(asr_processor.bos_audio_token)
                audio_end = prompt_text.index(asr_processor.eos_audio_token, audio_start)
                audio_tokens = "".join([asr_processor.audio_token] * audio_token_counts[audio_index])
                prompt_text = prompt_text[:audio_start] + audio_tokens + prompt_text[audio_end:]
                audio_index += 1

            assistant_start = prompt_text.find(asr_processor.assistant_token)
            if assistant_start < 0:
                raise ValueError("ASR prompt is missing the assistant token.")

            assistant_content_start = assistant_start + len(asr_processor.assistant_token)
            suffix_start = len(prompt_text)
            if prompt_text.endswith(asr_processor.user_token):
                suffix_start = len(prompt_text) - len(asr_processor.user_token)

            prefix_text = prompt_text[:assistant_content_start]
            content_text = prompt_text[assistant_content_start:suffix_start]
            suffix_text = prompt_text[suffix_start:]

            prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            content_ids = tokenizer(content_text, add_special_tokens=False)["input_ids"]
            suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
            row_ids = torch.tensor(prefix_ids + content_ids + suffix_ids, dtype=torch.long)

            input_rows.append(row_ids)
            attention_rows.append(torch.ones_like(row_ids, dtype=torch.long))

        if audio_index != len(audio_token_counts):
            raise ValueError("Unused audio token counts remained after building ASR prompts.")

        max_len = max(row.size(0) for row in input_rows)
        input_ids = torch.full((len(input_rows), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(input_rows), max_len), dtype=torch.long)

        for idx, row_ids in enumerate(input_rows):
            seq_len = row_ids.size(0)
            input_ids[idx, :seq_len] = row_ids
            attention_mask[idx, :seq_len] = attention_rows[idx]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "audios": audios}

    def _build_tts_sub_batch(self, tts_convs: list[list[dict[str, Any]]], tts_processor: Any) -> dict[str, torch.Tensor]:
        tokenizer = tts_processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0

        input_rows: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []

        for conv in tts_convs:
            template_text = tts_processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            start_idx = template_text.index(START_CONTENT_TAG) + len(START_CONTENT_TAG)
            end_idx = template_text.index(END_CONTENT_TAG, start_idx)

            prefix_text = template_text[:start_idx]
            content_text = template_text[start_idx:end_idx]
            suffix_text = template_text[end_idx:]

            prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            content_desc = encode_tts_content_text(tokenizer, content_text)
            suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
            row_ids = torch.tensor(prefix_ids + content_desc["input_ids"] + suffix_ids, dtype=torch.long)

            input_rows.append(row_ids)
            attention_rows.append(torch.ones_like(row_ids, dtype=torch.long))

        max_len = max(row.size(0) for row in input_rows)
        input_ids = torch.full((len(input_rows), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(input_rows), max_len), dtype=torch.long)

        for idx, row_ids in enumerate(input_rows):
            seq_len = row_ids.size(0)
            input_ids[idx, :seq_len] = row_ids
            attention_mask[idx, :seq_len] = attention_rows[idx]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _pad_2d(self, tensor: torch.Tensor, pad_value: int, target_len: int) -> torch.Tensor:
        if tensor.size(1) == target_len:
            return tensor
        pad = torch.full((tensor.size(0), target_len - tensor.size(1)), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        if batch_size == 0:
            raise ValueError("Empty batch")

        asr_indices: list[int] = []
        tts_indices: list[int] = []
        asr_features: list[dict[str, Any]] = []
        tts_features: list[dict[str, Any]] = []
        for idx, feature in enumerate(features):
            if self._get_task(feature) == "tts":
                tts_indices.append(idx)
                tts_features.append(feature)
            else:
                asr_indices.append(idx)
                asr_features.append(feature)

        sub_asr = None
        sub_tts = None
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id or 0

        if asr_features:
            asr_convs = [self._build_asr_conv(feature) for feature in asr_features]
            sub_asr = self._build_asr_sub_batch(asr_convs, self.processor)
            sub_asr["labels"] = self._make_labels_by_role_mask(
                sub_asr["input_ids"],
                sub_asr["attention_mask"],
                processor=self.processor,
            )

        if tts_features:
            active_tts_processor = self.tts_processor or self.processor
            tts_convs = self._build_tts_convs_from_features(tts_features)
            sub_tts = self._build_tts_sub_batch(tts_convs, active_tts_processor)
            sub_tts["labels"] = self._make_labels_by_role_mask(
                sub_tts["input_ids"],
                sub_tts["attention_mask"],
                processor=active_tts_processor,
            )

        max_len = 0
        if sub_asr is not None:
            max_len = max(max_len, sub_asr["input_ids"].size(1))
        if sub_tts is not None:
            max_len = max(max_len, sub_tts["input_ids"].size(1))

        device = sub_asr["input_ids"].device if sub_asr is not None else sub_tts["input_ids"].device
        final_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        final_attention = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        final_labels = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long, device=device)

        final_audios = None
        if sub_asr is not None and "audios" in sub_asr:
            asr_audios = sub_asr["audios"]
            final_audios = torch.zeros(
                (batch_size, asr_audios.size(1), asr_audios.size(2)),
                dtype=asr_audios.dtype,
                device=asr_audios.device,
            )

        if sub_asr is not None:
            padded_ids = self._pad_2d(sub_asr["input_ids"], pad_id, max_len)
            padded_attention = self._pad_2d(sub_asr["attention_mask"], 0, max_len)
            padded_labels = self._pad_2d(sub_asr["labels"], self.ignore_index, max_len)
            for row_idx, batch_idx in enumerate(asr_indices):
                final_input_ids[batch_idx] = padded_ids[row_idx]
                final_attention[batch_idx] = padded_attention[row_idx]
                final_labels[batch_idx] = padded_labels[row_idx]
                if final_audios is not None:
                    final_audios[batch_idx] = sub_asr["audios"][row_idx]

        if sub_tts is not None:
            padded_ids = self._pad_2d(sub_tts["input_ids"], pad_id, max_len)
            padded_attention = self._pad_2d(sub_tts["attention_mask"], 0, max_len)
            padded_labels = self._pad_2d(sub_tts["labels"], self.ignore_index, max_len)
            for row_idx, batch_idx in enumerate(tts_indices):
                final_input_ids[batch_idx] = padded_ids[row_idx]
                final_attention[batch_idx] = padded_attention[row_idx]
                final_labels[batch_idx] = padded_labels[row_idx]

        output = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention,
            "labels": final_labels,
        }
        if final_audios is not None:
            output["audios"] = final_audios
        return output