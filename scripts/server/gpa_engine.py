# -*- coding: utf-8 -*-
# Time      : 2025/12/23
# Author    : AutoArk-AI
# Desc      : Multifunctional Spark Engine (STT, TTS A, VC) - STT task supports dual Token input (GLM + Bicodec)

import sys
sys.path.append("..")
import asyncio
import math
import os.path
import re
from typing import Literal, Optional, List, Union, Dict, Any, AsyncIterator

import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, WhisperFeatureExtractor
import time
from loguru import logger

from base_engine import BaseEngine
from models.bicodec_tokenizer.spark_tokenizer import SparkTokenizer
from models.bicodec_tokenizer.spark_detokenizer import SparkDeTokenizer
from models.glm_speech_tokenizer.speech_token_extractor import SpeechTokenExtractor
from models.glm_speech_tokenizer.modeling_whisper import WhisperVQEncoder

class ArkGPAEngine(BaseEngine):
    SAMPLE_RATE = 16000
    AUDIO_FRAME_RATE = 50 

    def __init__(
            self,
            model_path: str,
            bicodec_audio_tokenizer_path: str,
            glm_speech_tokenizer_path: str,
            text_tokenizer_path: str,
            max_length: int = 4096,
            llm_device: str = "cuda",
            backend:  Literal["vllm", "llama-cpp", "sglang", "torch", "mlx-lm"] = "torch", 
            llm_attn_implementation: str = "sdpa",
            torch_dtype: str = "auto",
            llm_gpu_memory_utilization: float = 0.6,
            llm_batch_size: int = 20,
            seed: int = 0,
            glm_semantic_token_offset: int = 151727,
            bicodec_semantic_token_offset: int = 172207,
            bicodec_global_token_offset: int = 168111,
            **kwargs,
    ):
        self.seed = seed
        self.set_seed(seed)
        self.llm_device = llm_device
        self.glm_semantic_token_offset = glm_semantic_token_offset
        self.bicodec_semantic_token_offset = bicodec_semantic_token_offset
        self.bicodec_global_token_offset = bicodec_global_token_offset

        self.bicodec_tokenizer = SparkTokenizer(model_path=bicodec_audio_tokenizer_path, device=llm_device)
        self.bicodec_detokenizer = SparkDeTokenizer(model_path=bicodec_audio_tokenizer_path, device=llm_device)

        feature_extractor = WhisperFeatureExtractor.from_pretrained(glm_speech_tokenizer_path)
        audio_model = WhisperVQEncoder.from_pretrained(glm_speech_tokenizer_path).eval().to(llm_device)
        self.glm_tokenizer = SpeechTokenExtractor(model=audio_model, feature_extractor=feature_extractor, device=llm_device)

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, trust_remote_code=True)
        
        stop_tokens = ["<|im_end|>"]

        super().__init__(
            llm_model_path=model_path,
            max_length=max_length,
            llm_device=llm_device,
            backend=backend,
            llm_attn_implementation=llm_attn_implementation,
            torch_dtype=torch_dtype,
            llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            llm_batch_size=llm_batch_size,
            seed=seed,
            stop_tokens=stop_tokens,
            **kwargs
        )

    # ======================== Internal Helper Functions ========================

    async def _process_audio_input(self, audio: Union[str, torch.Tensor]) -> torch.Tensor:
        """Unified audio processing: load, resample, convert to mono, compress to 1D"""
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
            if sr != self.SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(wav)
        else:
            wav = audio

        # Convert to mono: [C, L] -> [L]
        if wav.ndim > 1:
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)
        return wav

    async def _get_global_tokens(self, audio: Union[str, torch.Tensor, None], is_zero: bool = False) -> torch.Tensor:
        if is_zero or audio is None:
            return torch.zeros((1, 1, 32), dtype=torch.long, device=self.llm_device)
        
        wav = await self._process_audio_input(audio)
        output = await self.bicodec_tokenizer.tokenize_async(wav)
        # output = self.bicodec_tokenizer.tokenize([audio])
        # return output['global_tokens'].to(self.llm_device)
        return output['feature']['global_tokens'].to(self.llm_device)

    # async def _get_glm_tokens(self, audio: Union[str, torch.Tensor]) -> List[int]:
    #     # wav = await self._process_audio_input(audio)
    #     # output = await self.glm_tokenizer.extract_async(wav)
    #     output = self.glm_tokenizer.extract([audio])
    #     return [int(t) + self.glm_semantic_token_offset for t in output]
    
    async def _get_glm_tokens(self, audio: Union[str, torch.Tensor]) -> List[int]:
        # output 是 List[List[int]]，例如 [[101, 102, ...]]
        # output = self.glm_tokenizer.extract([audio])
        
        # if not output or len(output) == 0:
        #     return []

        # # 1. 取出第一条数据的 token 列表
        # tokens_list = output[0] 

        # # 2. 转为 Tensor 进行矢量加法 (这样就可以直接 + offset 了)
        # # 注意：如果不指定 device，默认在 CPU，对于这种简单的加法通常比搬运到 GPU 更快
        # tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
        
        # # 3. 加 offset 并转回 list
        # result_tensor = tokens_tensor + self.glm_semantic_token_offset
        
        # return result_tensor.tolist()
        wav = await self._process_audio_input(audio)
        output = await self.glm_tokenizer.extract_async(wav)
        return [int(t) + self.glm_semantic_token_offset for t in output]

    async def _get_bicodec_semantic_tokens(self, audio: Union[str, torch.Tensor]) -> List[int]:
        """Get Bicodec semantic token list and add offset"""
        output = self.bicodec_tokenizer.tokenize([audio])
        tokens = output['semantic_tokens'].flatten().tolist()
        return [int(t) + self.bicodec_semantic_token_offset for t in tokens]

    async def _tokens2wav(self, global_tokens: torch.Tensor, semantic_tokens: List[int]) -> np.ndarray:
        if not semantic_tokens: return np.array([], dtype=np.float32)
        sem_tensor = torch.tensor(semantic_tokens, dtype=torch.long, device=self.llm_device).unsqueeze(0)
        req = {
            "global_tokens": global_tokens.squeeze(1),
            "semantic_tokens": sem_tensor.squeeze(0),
        }
        out = await self.bicodec_detokenizer.detokenize_async(req)
        audio = out['audio'][0].cpu().numpy().astype(np.float32)
        audio -= np.mean(audio)
        return np.clip(audio, -0.995, 0.995)

    # ======================== Functional Functions (STT Update) ========================

    async def _build_stt_prompt(self, audio: Union[str, torch.Tensor]) -> str:
        """构建 STT 任务的 Prompt：包含 GLM Token 和 Bicodec Semantic Token"""
        # 1. 获取音频 Tokens
        glm_tokens = await self._get_glm_tokens(audio)
        bicodec_tokens = await self._get_bicodec_semantic_tokens(audio)

        # 2. 获取特殊 Token 的 ID (关键：必须使用 add_special_tokens=False)
        #    这样可以防止 tokenizer 自动插入 BOS/EOS 等破坏结构的标记
        t_start_glm = self.text_tokenizer.encode("<|start_glm_token|>", add_special_tokens=False)
        t_end_glm = self.text_tokenizer.encode("<|end_glm_token|>", add_special_tokens=False)
        t_start_sem = self.text_tokenizer.encode("<|start_semantic_token|>", add_special_tokens=False)
        t_end_sem = self.text_tokenizer.encode("<|end_semantic_token|>", add_special_tokens=False)
        t_start_con = self.text_tokenizer.encode("<|start_content|>", add_special_tokens=False)

        # 4. 拼接 ID 序列
        input_ids = (
            t_start_glm +
            glm_tokens +
            t_end_glm +
            t_start_sem +
            bicodec_tokens +
            t_end_sem +
            t_start_con
         
        )
        
        # 5. 解码为字符串 Prompt
        # 注意：这里假设 tokenizer 能正确地将 token ID 映射回 <|speech_x|> 这种字符串形式
        # 如果 vllm/backend 接受 token_ids，直接传 ids 更安全，但为了兼容 BaseEngine 接口这里转回 str
        result = self.text_tokenizer.decode(input_ids)
        # print(f"stt input is {result}")
        return result

    async def stt_async(self, audio: Union[str, torch.Tensor], **kwargs) -> str:
        prompt = await self._build_stt_prompt(audio)
        return await self.generator.async_generate(prompt=prompt, **kwargs)

    async def stt_stream_async(self, audio: Union[str, torch.Tensor], **kwargs) -> AsyncIterator[str]:
        prompt = await self._build_stt_prompt(audio)
        async for chunk in self.generator.async_stream_generate(prompt=prompt, **kwargs):
            yield chunk

    # ======================== Other TTS & VC Functions (Keep Unchanged) ========================

    async def tts_a_async(self, text: str, ref_audio: Union[str, torch.Tensor], **kwargs) -> np.ndarray:
        global_tokens = await self._get_global_tokens(ref_audio)
        prompt = self._build_tts_prompt(text, global_tokens)
        return await self._voice_full_gen(prompt, global_tokens, **kwargs)

    async def tts_a_stream_async(self, text: str, ref_audio: Union[str, torch.Tensor], **kwargs) -> AsyncIterator[np.ndarray]:
        global_tokens = await self._get_global_tokens(ref_audio)
        prompt = self._build_tts_prompt(text, global_tokens)
        async for chunk in self._voice_stream_gen(prompt, global_tokens, **kwargs):
            yield chunk

    async def vc_async(self, src_audio: Union[str, torch.Tensor], ref_audio: Union[str, torch.Tensor], **kwargs) -> np.ndarray:
        global_tokens_ref = await self._get_global_tokens(ref_audio)
        prompt = await self._build_vc_prompt(src_audio, global_tokens_ref)
        return await self._voice_full_gen(prompt, global_tokens_ref, **kwargs)

    async def vc_stream_async(self, src_audio: Union[str, torch.Tensor], ref_audio: Union[str, torch.Tensor], **kwargs) -> AsyncIterator[np.ndarray]:
        global_tokens_ref = await self._get_global_tokens(ref_audio)
        prompt = await self._build_vc_prompt(src_audio, global_tokens_ref)
        async for chunk in self._voice_stream_gen(prompt, global_tokens_ref, **kwargs):
            yield chunk

    # ======================== Core Generation Logic ========================

    def _build_tts_prompt(self, text, global_tokens):
        g_list = (global_tokens.flatten() + self.bicodec_global_token_offset).tolist()
        g_str = f"<|start_global_token|>{self.text_tokenizer.decode(g_list)}<|end_global_token|>"
        return f"{g_str}<|start_content|>{text}<|end_content|>"

    async def _build_vc_prompt(self, src_audio, global_tokens_ref):
        # src_wav = await self._process_audio_input(src_audio)
        # src_res = await self.bicodec_tokenizer.tokenize_async(src_wav)
        src_res = self.bicodec_tokenizer.tokenize([src_audio])
        sem_src = (src_res['semantic_tokens'].flatten() + self.bicodec_semantic_token_offset).tolist()
        g_ref_list = (global_tokens_ref.flatten() + self.bicodec_global_token_offset).tolist()
        
        return (f"<|start_global_token|>{self.text_tokenizer.decode(g_ref_list)}<|end_global_token|>"
                f"<|start_semantic_token|>{self.text_tokenizer.decode(sem_src)}<|end_semantic_token|>"
                f"<|end_content|>")

    async def _voice_full_gen(self, prompt, global_tokens, **kwargs) -> np.ndarray:
        full_text = await self.generator.async_generate(prompt=prompt, **kwargs)
        ids = [int(x) for x in re.findall(r"<\|bicodec_semantic_(\d+)\|>", full_text)]
        wav = await self._tokens2wav(global_tokens, ids)
        return (wav * 32767).astype(np.int16)

    async def _voice_stream_gen(self, prompt, global_tokens, 
                                audio_chunk_duration=1.0, 
                                audio_chunk_overlap_duration=0.001, 
                                **kwargs) -> AsyncIterator[np.ndarray]:
        chunk_size = math.ceil(audio_chunk_duration * self.AUDIO_FRAME_RATE)
        overlap_size = math.ceil(audio_chunk_overlap_duration * self.AUDIO_FRAME_RATE)
        xfade_len = int(audio_chunk_overlap_duration * self.SAMPLE_RATE)
        
        fade_in = np.linspace(0, 1, xfade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, xfade_len, dtype=np.float32)

        buff = []
        last_chunk_audio = None

        async for chunk_text in self.generator.async_stream_generate(prompt=prompt, **kwargs):
            new_ids = re.findall(r"<\|bicodec_semantic_(\d+)\|>", chunk_text)
            for tid in new_ids:
                buff.append(int(tid))
                if len(buff) >= chunk_size:
                    chunk_wav = await self._tokens2wav(global_tokens, buff[:chunk_size])
                    xf = min(xfade_len, len(chunk_wav)) if last_chunk_audio is None else min(xfade_len, len(chunk_wav), len(last_chunk_audio))
                    
                    if last_chunk_audio is None:
                        atk = max(1, min(int(0.01 * self.SAMPLE_RATE), len(chunk_wav)//8))
                        chunk_wav[:atk] *= np.linspace(0, 1, atk, dtype=np.float32)
                        yield_wav = chunk_wav[:-xf] if xf > 0 else chunk_wav
                    else:
                        overlap = chunk_wav[:xf] * fade_in[:xf] + last_chunk_audio[-xf:] * fade_out[:xf]
                        yield_wav = np.concatenate([overlap, chunk_wav[xf:]], axis=0)
                    
                    yield (yield_wav * 32767).astype(np.int16)
                    last_chunk_audio = chunk_wav
                    buff = buff[chunk_size - overlap_size:]

        if buff:
            tail_wav = await self._tokens2wav(global_tokens, buff)
            if last_chunk_audio is not None:
                xf = min(xfade_len, len(tail_wav), len(last_chunk_audio))
                overlap = tail_wav[:xf] * fade_in[:xf] + last_chunk_audio[-xf:] * fade_out[:xf]
                out = np.concatenate([overlap, tail_wav[xf:]], axis=0)
            else:
                out = tail_wav
            
            rel = min(int(0.01 * self.SAMPLE_RATE), len(out))
            out[-rel:] *= np.linspace(1, 0, rel, dtype=np.float32)
            yield (out * 32767).astype(np.int16)