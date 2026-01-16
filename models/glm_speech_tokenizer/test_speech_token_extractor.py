

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append("../../..")
import asyncio
import time
from datetime import datetime

import torch
import torchaudio
from transformers import WhisperFeatureExtractor
from arktts.models.glm_speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_token_extractor import SpeechTokenExtractor  # 你实现的类
_RESAMPLE_CACHE: dict[int, torchaudio.transforms.Resample] = {}

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def sync_cuda(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)

def load_wav_as_tuple(path: str,target_sr: int = 16000):
    """读取 wav -> (mono_waveform_1d, sample_rate)；保持在CPU上交给 extractor 处理。"""
    wav, sr = torchaudio.load(path)  # [C, T]
    
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0)        # -> [T] 变单声道
    else:
        wav = wav.squeeze(0)         # [1, T] -> [T]
    # 保证是连续的 float32（特征器吃 numpy.float32 会更快）
    wav = wav.contiguous().to(torch.float32).cpu()
    if sr != target_sr:
        if sr not in _RESAMPLE_CACHE:
            _RESAMPLE_CACHE[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
        wav = _RESAMPLE_CACHE[sr](wav.unsqueeze(0)).squeeze(0)
        sr = target_sr
    
    # print(f"type wave is {type(wav)}")
    return wav

async def main():
    # --- 1️⃣ 路径配置 ---
    MODEL_PATH = "/data/yumu/model/glm-4-voice-tokenizer"
    AUDIO_PATH1 = "/data/yumu/data/audio_data/qiduoduo_tts_out/00000013.wav"
    AUDIO_PATH2 = "/data/yumu/data/audio_data/qiduoduo_tts_out/00000012.wav"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert os.path.exists(AUDIO_PATH1), f"音频文件不存在: {AUDIO_PATH1}"
    assert os.path.exists(MODEL_PATH), f"模型路径不存在: {MODEL_PATH}"

    print(f"[{ts()}] 启动测试")
    print(f"  - DEVICE        : {DEVICE}")
    print(f"  - MODEL_PATH    : {MODEL_PATH}")
    print(f"  - AUDIO1        : {AUDIO_PATH1}")
    print(f"  - AUDIO2        : {AUDIO_PATH2 if os.path.exists(AUDIO_PATH2) else '(不存在，将重复 AUDIO1)'}")

    # --- 2️⃣ 先把音频读入内存（改动点）---
    audio1 = load_wav_as_tuple(AUDIO_PATH1)
    audio2 = load_wav_as_tuple(AUDIO_PATH2) if os.path.exists(AUDIO_PATH2) else audio1

    # --- 3️⃣ 加载模型与特征提取器 ---
    print(f"\n[{ts()}] 加载 WhisperVQ 模型与特征提取器中...")
    t0 = time.perf_counter()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH)

    model = WhisperVQEncoder.from_pretrained(MODEL_PATH).eval().to(DEVICE)
    if DEVICE.startswith("cuda"):
        model = model.half()  # 半精度仅保留一次
    sync_cuda(DEVICE)
    t1 = time.perf_counter()
    print(f"[{ts()}] 模型加载完成，用时 {(t1 - t0)*1000:.1f} ms")

    # --- 4️⃣ 初始化提取器 ---
    t0 = time.perf_counter()
    extractor = SpeechTokenExtractor(
        model=model,
        feature_extractor=feature_extractor,
        device=DEVICE,
        batch_size=400,
        wait_timeout=0.01,
    )
    sync_cuda(DEVICE)
    t1 = time.perf_counter()
    print(f"[{ts()}] ✅ SpeechTokenExtractor 初始化完成，用时 {(t1 - t0)*1000:.1f} ms")

    # --- 5️⃣ 同步测试（传入预加载的 (wav, sr) 元组）---
    print(f"\n[{ts()}] [同步模式] extract() 开始")
    t0 = time.perf_counter()
    sync_tokens_list = extractor.extract([audio1])  # ★ 改：不再传路径
    sync_cuda(DEVICE)
    t1 = time.perf_counter()
    sync_tokens = sync_tokens_list[0]
    print(f"[{ts()}] [同步模式] 完成：{len(sync_tokens)} tokens")
    print(f"  - 预览：{sync_tokens[:20]} ...")
    print(f"  - 耗时：{(t1 - t0)*1000:.1f} ms  （单样本）")

    # --- 6️⃣ 异步测试（同样传入元组）---
    print(f"\n[{ts()}] [异步模式] extract_async() 并发开始")

    async def async_worker(audio_utt):
        t_a0 = time.perf_counter()
        print(f"type audio_utt is {type(audio_utt)}")
        tokens = await extractor.extract_async(audio_utt)  # ★ 改：不再传路径
        sync_cuda(DEVICE)
        t_a1 = time.perf_counter()
        print(f"  · → {len(tokens)} tokens, {(t_a1 - t_a0)*1000:.1f} ms")
        return tokens, (t_a1 - t_a0)

    # 这里保持你原本的 20+20 并发规模，只是把对象换成内存元组
    test_inputs = [audio1] * 2 + [audio2] * 2

    t0 = time.perf_counter()
    results = await asyncio.gather(*(async_worker(aud) for aud in test_inputs))
    sync_cuda(DEVICE)
    t1 = time.perf_counter()

    per_req_ms = [dt * 1000 for _, dt in results]
    all_tokens = [tokens for tokens, _ in results]

    print(f"[{ts()}] [异步模式] 完成")
    print(f"  - 总请求数：{len(results)}")
    print(f"  - 总耗时  ：{(t1 - t0)*1000:.1f} ms")
    print(f"  - 单请求耗时（ms）：{[round(x,1) for x in per_req_ms]}")
    print(f"  - 平均单请求耗时：{(sum(per_req_ms)/len(per_req_ms)):.1f} ms")
    print(f"  - 任一结果预览  ：{all_tokens[0][:10]}")
    print(f"\n[{ts()}] ✅ 所有测试完成。")

if __name__ == "__main__":
    asyncio.run(main())
