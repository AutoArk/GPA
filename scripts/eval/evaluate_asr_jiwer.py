#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import asyncio
import io
import wave
import aiohttp
import numpy as np
import librosa
import re
import jiwer
import uuid
from tqdm import tqdm
import torch
import torchaudio

# ------------------ 配置 ------------------
# 输入文件路径
INPUT_JSONL = "/data/yumu/arkasr/eval/test_net_100.jsonl"
# "/data/yumu/data/audio_data/WenetSpeech/wenetspeech_segments_test_net_simplify.jsonl"
# "/data/yumu/data/audio_data/WenetSpeech/audio/test_meeting/wenetspeech_segments_audio_text.jsonl"
RESULT_JSONL = "wenetspeech_asr_eval_http.jsonl"

# HTTP 服务地址
STT_URL = os.getenv("ARKASR_HTTP_URL", "http://localhost:8000/stt")

TARGET_SR = 16000
BATCH_SIZE = 20
CONCURRENCY = 20  # 如果服务端显存吃紧，请降低此数值 (如 1-5)
WRITE_BUFFER = 10 

# === 参数配置 ===
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.0

# ------------------ 文本清洗 ------------------
_PUNCT_PATTERN = re.compile(
    r"[，。！？、,.!?;:\"'“”‘’（）()【】\[\]<>《》{}…—\-~·`@#$%^&*_+=|\\/]"
)

# 特殊 Token 正则
_SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|(?:"
    r"bicodec_(?:semantic|global)_\d+|"                  
    r"(?:start|end)_(?:global_token|glm_token|semantic_token|content)"
    r")\|>"
)

def remove_special_tokens(text: str) -> str:
    """去除模型输出的所有特殊控制 token"""
    if not text:
        return ""
    text = _SPECIAL_TOKEN_PATTERN.sub("", text)
    return text.strip()

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = _PUNCT_PATTERN.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def contains_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", s))

def normalize_for_wer(s: str) -> str:
    s = clean_text(s).lower()
    if contains_cjk(s):
        chars = [ch for ch in s if not ch.isspace()]
        return " ".join(chars)
    return re.sub(r"\s+", " ", s).strip()

# ------------------ HTTP 请求逻辑 ------------------

def numpy_to_wav_bytes(audio_int16: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()

async def http_infer(session: aiohttp.ClientSession, audio_int16: np.ndarray) -> str:
    wav_data = numpy_to_wav_bytes(audio_int16, TARGET_SR)
    data = aiohttp.FormData()
    filename = f"audio_{uuid.uuid4().hex}.wav"
    data.add_field('file', wav_data, filename=filename, content_type='audio/wav')
    data.add_field('temperature', str(TEMPERATURE))
    data.add_field('repetition_penalty', str(REPETITION_PENALTY))

    try:
        # 建议加上 Connection: close 防止长连接导致的偶发错误
        async with session.post(STT_URL, data=data, headers={"Connection": "close"}) as response:
            if response.status == 200:
                # 尝试解析 JSON，如果失败则回退到 text (兼容旧版接口)
                try:
                    res_json = await response.json()
                    return res_json.get("text", "")
                except:
                    return await response.text()
            else:
                return ""
    except Exception as e:
        print(f"Request failed: {e}")
        return ""

# ------------------ 音频统一化 ------------------
def to_16k_mono_int16(audio: np.ndarray, sr: int):
    if audio.ndim == 2: audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32767.0
    sg = torch.from_numpy(audio).unsqueeze(0)
    sg = torchaudio.functional.resample(sg, sr, TARGET_SR)
    sg = sg.clamp_(-1, 1)
    return (sg.squeeze(0).numpy() * 32767).astype(np.int16)

# ------------------ JSONL Loader ------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ------------------ 主逻辑 ------------------
async def main():
    if not os.path.exists(INPUT_JSONL):
        print(f"Error: 输入文件不存在: {INPUT_JSONL}")
        return

    data = list(load_jsonl(INPUT_JSONL))
    total = len(data)
    print(f"✅ Loaded {total:,} samples")
    print(f"⚙️ Config: Temperature={TEMPERATURE}, RepetitionPenalty={REPETITION_PENALTY}")

    write_q = asyncio.Queue()
    async def writer_task():
        with open(RESULT_JSONL, "w", encoding="utf-8") as f:
            while True:
                r = await write_q.get()
                if r is None: break
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                write_q.task_done()
    writer = asyncio.create_task(writer_task())

    sem = asyncio.Semaphore(CONCURRENCY)
    
    # 统计变量
    n_ok = 0
    total_errors_wer = 0.0  # 累积 WER 错误数 (distance)
    total_errors_cer = 0.0  # 累积 CER 错误数 (distance)
    total_ref_len = 0       # 累积参考文本长度 (denom)

    timeout = aiohttp.ClientTimeout(total=300, connect=60) # 增加超时时间
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        async def process(item):
            async with sem:
                ap = item["audio"]
                ref = item.get("text", "")
                b = float(item.get("begin_time", -1))
                e = float(item.get("end_time", -1))
                duration = None if e < 0 else max(0, e - b)

                try:
                    audio, sr = librosa.load(ap, offset=b if b>=0 else None,
                                            duration=duration, sr=None, mono=False)
                    audio16 = to_16k_mono_int16(audio, sr)
                    hyp_raw = await http_infer(session, audio16)
                except Exception as e:
                    print(f"Error processing {ap}: {e}")
                    hyp_raw = ""
            
            hyp_pure = remove_special_tokens(hyp_raw)
            
            ref_clean = clean_text(ref)
            hyp_clean = clean_text(hyp_pure)
            
            # 计算当前样本的长度
            curr_len = len(list(ref_clean))
            if curr_len == 0: curr_len = 1 # 防止除以0

            # jiwer.cer/wer 返回的是比率 (Rate)，乘以长度得到具体的错误数 (Errors)
            cer_errors = jiwer.cer(ref_clean, hyp_clean) * curr_len
            wer_errors = jiwer.wer(normalize_for_wer(ref_clean),
                                   normalize_for_wer(hyp_clean)) * curr_len

            r = {
                "audio": ap,
                "ref_text": ref,
                "pred_text": hyp_pure,
                "ref_text_clean": ref_clean,
                "pred_text_clean": hyp_clean,
                "cer_errors": cer_errors, # 存具体错误数
                "wer_errors": wer_errors, # 存具体错误数
                "len": curr_len,
            }
            return r

        # 使用 tqdm 显示进度
        with tqdm(total=total, desc="ASR Eval", unit="utt") as pbar:
            for batch in chunked(data, BATCH_SIZE):
                tasks = [asyncio.create_task(process(x)) for x in batch]
                
                for fut in asyncio.as_completed(tasks):
                    r = await fut
                    await write_q.put(r)
                    
                    n_ok += 1
                    total_errors_wer += r["wer_errors"]
                    total_errors_cer += r["cer_errors"]
                    total_ref_len += r["len"]
                    
                    pbar.update(1)
                    
                    # === 实时更新进度条后缀 ===
                    if total_ref_len > 0:
                        curr_wer = total_errors_wer / total_ref_len
                        curr_cer = total_errors_cer / total_ref_len
                        pbar.set_postfix({
                            "WER": f"{curr_wer:.2%}",
                            "CER": f"{curr_cer:.2%}"
                        })

    await write_q.put(None)
    await writer

    final_len = max(1, total_ref_len)
    print(f"\n✅ Done {n_ok} samples:")
    print(f"   Final WER: {total_errors_wer/final_len*100:.2f}%")
    print(f"   Final CER: {total_errors_cer/final_len*100:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())