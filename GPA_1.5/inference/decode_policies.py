import re
from typing import Any, List, Optional

import torch
from transformers.generation.logits_process import LogitsProcessor


CONTROL_TOKEN_PATTERN = re.compile(r"^<.*>$")


class BlockTokenIdsFromLogitsProcessor(LogitsProcessor):
    def __init__(self, block_from_id: Optional[int], block_token_ids: Optional[List[int]] = None):
        self.block_from_id = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
        self.block_token_ids = sorted(set(int(token_id) for token_id in (block_token_ids or [])))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        if self.block_from_id is not None and self.block_from_id < vocab_size:
            scores[:, self.block_from_id:] = -float("inf")
        if self.block_token_ids:
            valid_token_ids = [token_id for token_id in self.block_token_ids if 0 <= token_id < vocab_size]
            if valid_token_ids:
                scores[:, valid_token_ids] = -float("inf")
        return scores


def normalize_token_ids(token_ids: Any) -> List[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        return [int(token_id) for token_id in token_ids if token_id is not None]
    return [int(token_ids)]


def build_asr_keep_token_ids(model, tokenizer) -> List[int]:
    keep_token_ids = set()
    keep_token_ids.update(normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    keep_token_ids.update(normalize_token_ids(getattr(getattr(model, "config", None), "eos_token_id", None)))
    keep_token_ids.update(
        normalize_token_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    )
    return sorted(keep_token_ids)


def build_asr_extra_block_token_ids(
    tokenizer,
    keep_token_ids: Optional[List[int]] = None,
    block_from_id: Optional[int] = None,
) -> List[int]:
    keep_token_ids = set(int(token_id) for token_id in (keep_token_ids or []))
    max_control_token_id = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)

    block_token_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    for token_id, token_meta in added_tokens_decoder.items():
        token_id = int(token_id)
        if max_control_token_id is not None and token_id >= max_control_token_id:
            continue
        token_content = getattr(token_meta, "content", None)
        if token_content is None and isinstance(token_meta, dict):
            token_content = token_meta.get("content")
        if token_content and CONTROL_TOKEN_PATTERN.match(token_content):
            block_token_ids.add(token_id)

    block_token_ids.difference_update(keep_token_ids)
    return sorted(block_token_ids)


def strip_last_if_stop(generated_ids: torch.Tensor, stop_id: int) -> torch.Tensor:
    if generated_ids.numel() == 0:
        return generated_ids
    if int(generated_ids[-1].item()) == int(stop_id):
        return generated_ids[:-1]
    return generated_ids