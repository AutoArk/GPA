from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from tts_han_char_tokenizer import encode_tts_content_text

from .decode_policies import strip_last_if_stop
from .prompts import END_CONTENT_TAG, START_CONTENT_TAG, build_tts_conversation, extract_semantic_ids_from_text


def build_tts_inputs(tts_processor, conversation: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    tokenizer = tts_processor.tokenizer
    template_text = tts_processor.apply_chat_template(
        conversation,
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


@torch.no_grad()
def run_tts(
    *,
    model,
    tts_processor,
    tokenizer,
    spark_tokenizer,
    spark_detokenizer,
    ref_audio_path: str,
    text: str,
    tts_stop_id: int = 151665,
    max_new_tokens: int = 4096,
    max_semantic_tokens: int = 600,
    device: str = "cpu",
) -> Tuple[Union[torch.Tensor, np.ndarray], List[int], str]:
    reference_output = spark_tokenizer.tokenize([ref_audio_path])
    global_tokens = reference_output["global_tokens"]
    global_token_ids = global_tokens[0, 0].detach().cpu().tolist()

    conversation = build_tts_conversation(text=text, global_token_ids=global_token_ids)
    inputs = build_tts_inputs(tts_processor, conversation)
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    prompt_length = inputs["input_ids"].shape[1]
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        repetition_penalty=1.1,
        temperature=0.3,
    )
    generated_ids = strip_last_if_stop(generated[0, prompt_length:], tts_stop_id)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    semantic_ids = extract_semantic_ids_from_text(generated_text)
    if not semantic_ids:
        tokens = tokenizer.convert_ids_to_tokens(generated_ids[:200].tolist())
        raise RuntimeError(
            "No semantic speech tokens were generated. "
            f"First generated tokens: {tokens}"
        )
    semantic_ids = semantic_ids[:max_semantic_tokens]

    semantic_tokens = torch.tensor(semantic_ids, dtype=torch.long).unsqueeze(0)
    waveform = spark_detokenizer.detokenize(
        semantic_tokens=semantic_tokens,
        global_tokens=global_tokens,
    )
    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu().squeeze().float().numpy()
    return waveform, semantic_ids, generated_text