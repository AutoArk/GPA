import re
from typing import Any, Dict, Optional

import torch
from transformers.generation.logits_process import LogitsProcessorList

from .decode_policies import (
    BlockTokenIdsFromLogitsProcessor,
    build_asr_extra_block_token_ids,
    build_asr_keep_token_ids,
)
from .prompts import build_asr_conversation


TRAILING_ROLE_TOKEN_RE = re.compile(r"(?:\s*<\|(user|assistant|system)\|>)+\s*$")


def as_dict(batch: Any) -> Dict[str, Any]:
    if isinstance(batch, dict):
        return batch
    if hasattr(batch, "keys") and hasattr(batch, "__getitem__"):
        return {key: batch[key] for key in batch.keys()}
    raise TypeError(f"Unexpected inputs type: {type(batch)}")


def normalize_asr_text(text: str) -> str:
    normalized = text.strip()
    normalized = TRAILING_ROLE_TOKEN_RE.sub("", normalized).strip()
    return normalized


@torch.no_grad()
def run_asr(
    *,
    model,
    processor,
    tokenizer,
    audio_path: str,
    begin_time: float = -1,
    end_time: float = -1,
    sampling_rate: int = 16000,
    max_audio_seconds: int = 30,
    max_new_tokens: int = 256,
    asr_block_token_id_from: Optional[int] = 151670,
    device: str = "cpu",
) -> str:
    conversation = build_asr_conversation(audio_path=audio_path, begin_time=begin_time, end_time=end_time)
    audio_max_length = int(max_audio_seconds * sampling_rate)

    inputs_raw = processor.apply_chat_template(
        conversation,
        return_tensors="pt",
        sampling_rate=sampling_rate,
        audio_padding="longest",
        add_generation_prompt=True,
        text_kwargs={"padding": "longest"},
        audio_max_length=audio_max_length,
    )

    if torch.is_tensor(inputs_raw):
        raise RuntimeError(
            "ASR apply_chat_template returned a tensor without encoded audio. "
            f"processor={processor.__class__}"
        )

    inputs = as_dict(inputs_raw)
    if "audios" not in inputs:
        raise RuntimeError(
            "ASR inputs are missing the 'audios' field, so the model did not receive audio features. "
            f"processor={processor.__class__}, keys={list(inputs.keys())}"
        )

    if "attention_mask" not in inputs and "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

    model_dtype = next(model.parameters()).dtype
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                inputs[key] = value.to(device=device, dtype=model_dtype)
            else:
                inputs[key] = value.to(device)

    prompt_length = inputs["input_ids"].shape[1]
    keep_token_ids = build_asr_keep_token_ids(model, tokenizer)
    extra_block_token_ids = build_asr_extra_block_token_ids(
        tokenizer,
        keep_token_ids=keep_token_ids,
        block_from_id=asr_block_token_id_from,
    )

    logits_processor = None
    if (asr_block_token_id_from is not None and int(asr_block_token_id_from) >= 0) or extra_block_token_ids:
        logits_processor = LogitsProcessorList(
            [
                BlockTokenIdsFromLogitsProcessor(
                    block_from_id=asr_block_token_id_from,
                    block_token_ids=extra_block_token_ids,
                )
            ]
        )

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        repetition_penalty=1.0,
        temperature=0.1,
        logits_processor=logits_processor,
    )
    generated_ids = generated[0, prompt_length:]
    return normalize_asr_text(tokenizer.decode(generated_ids))