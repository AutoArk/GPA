# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:57
# Author    :Hui Huang
from threading import Thread
from typing import Optional, Literal, AsyncIterator, List

import torch

from .base_llm import BaseLLM
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList)
import uuid

__all__ = ["TorchGenerator"]


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
        self.stop = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        for i, input_id in enumerate(input_ids):
            if i >= len(self.stop):
                self.stop.append(False)

            if input_id[-1] in self.stop_token_ids:
                self.stop[i] = True
            if self.stop[i]:
                input_ids[i][-1] = self.stop_token_ids[0]

        if all(self.stop):
            self.stop = []
            return True
        return False


class TorchGenerator(BaseLLM):
    def __init__(self,
                 model_path: str,
                 max_length: int = 32768,
                 device: str = "cpu",
                 attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
                 torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
                 cache_implementation: Optional[str] = None,
                 stop_tokens: Optional[list[str]] = None,
                 stop_token_ids: Optional[List[int]] = None,
                 **kwargs):
        self.device = torch.device(device)
        self.cache_implementation = cache_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, torch_dtype, "auto"),
            attn_implementation=attn_implementation,
            **kwargs
        )
        self.model.eval().to(self.device)

        self.streamer: dict[str, TextIteratorStreamer] = {}

        super(TorchGenerator, self).__init__(
            tokenizer=model_path,
            max_length=max_length,
            stop_tokens=stop_tokens,
            stop_token_ids=stop_token_ids,
        )

    async def async_generate(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs
    ) -> str:
        """
        如果使用动态批处理，hf代码无法为每一个请求单独设置generation 参数，所以还是逐个处理吧。
        如果想要动态批处理，建议使用vllm、sglang
        Args:
            prompt:
            max_tokens:
            temperature:
            top_p:
            top_k:
            **kwargs:

        Returns:

        """
        max_tokens = self.valid_max_tokens(max_tokens)
        input_ids = self.tokenize(prompt, max_tokens)

        input_ids = torch.LongTensor([input_ids]).to(self.device)
        generated_ids = self.model.generate(
            input_ids,
            generation_config=GenerationConfig(
                max_length=self.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                # do_sample=True,
                cache_implementation=self.cache_implementation,
                **kwargs
            ),
        )
        prompt_length = input_ids.size(1)
        completion_ids = generated_ids[:, prompt_length:]
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return completions_text[0]

    async def async_stream_generate(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs) -> AsyncIterator[str]:
        max_tokens = self.valid_max_tokens(max_tokens)
        input_ids = self.tokenize(prompt, max_tokens)

        input_ids = torch.LongTensor([input_ids]).to(self.device)
        request_id = str(uuid.uuid4().hex)

        # 避免并发请求时，streamer错乱
        self.streamer[request_id] = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True)
        cur_streamer = self.streamer[request_id]
        stop_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=cur_streamer,
            generation_config=GenerationConfig(
                max_length=self.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                cache_implementation=self.cache_implementation,
                **kwargs
            ),
            use_cache=True,
            stopping_criteria=stop_criteria)
        Thread(target=self.model.generate, kwargs=generation_kwargs).start()
        for token in cur_streamer:
            yield token

        self.streamer.pop(request_id)
