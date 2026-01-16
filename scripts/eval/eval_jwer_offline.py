#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import uuid
import jiwer
import torch
import librosa
import soundfile as sf  # 需要安装: pip install soundfile
import numpy as np
from tqdm import tqdm

# ------------------ 路径设置 (请根据实际情况调整) ------------------
# 添加项目根目录到路径以便导入 gpa_inference
sys.path.append("../../") 

try:
    from scripts.inference.gpa_inference import GPAInference
except ImportError:
    print("错误: 无法导入 GPAInference，请检查 sys.path.append 的路径是否正确指向包含 gpa_inference.py 的目录")
    sys.exit(1)

# === 模型路径配置 ===
TOKENIZER_PATH = "/data/yumu/model/glm-4-voice-tokenizer"
TEXT_TOKENIZER_PATH = "/data/yumu/model/ark_audio_v1_0_3_b"
BICODEC_TOKENIZER_PATH = "/data/arki_production/model/SparkAudio/Spark-TTS-0___5B/"
GPA_MODEL_PATH = "/data/yumu/model/ark_audio_v1_0_3_b"

# === 评测文件配置 ===
INPUT_JSONL = "test.jsonl"
RESULT_JSONL = "wenetspeech_asr_eval_offline.jsonl" # 输出文件名改一下，区分http版
OUTPUT_DIR = "."

# === 其他配置 ===
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000 # 这里保持 16k，如果模型需要其他采样率，模型内部通常会处理，但切片保存最好统一
TEMP_WAV_DIR = "./temp_wavs" # 用于存放临时切片音频的目录

# ------------------ 文本清洗 (保持原逻辑不变) ------------------
_PUNCT_PATTERN = re.compile(
    r"[，。！？、,.!?;:\"'“”‘’（）()【】\[\]<>《》{}…—\-~·`@#$%^&*_+=|\\/]"
)

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

# ------------------ JSONL 工具 ------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)

# ------------------ 主逻辑 ------------------
def main():
    if not os.path.exists(INPUT_JSONL):
        print(f"Error: 输入文件不存在: {INPUT_JSONL}")
        return
    
    # 确保临时目录存在
    os.makedirs(TEMP_WAV_DIR, exist_ok=True)

    # 1. 初始化模型
    print(f"正在加载模型到 {DEVICE} ...")
    try:
        inference = GPAInference(
            tokenizer_path=TOKENIZER_PATH,
            text_tokenizer_path=TEXT_TOKENIZER_PATH,
            bicodec_tokenizer_path=BICODEC_TOKENIZER_PATH,
            gpa_model_path=GPA_MODEL_PATH,
            output_dir=OUTPUT_DIR,
            device=DEVICE
        )
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    data = list(load_jsonl(INPUT_JSONL))
    total = len(data)
    print(f"✅ Loaded {total:,} samples")

    # 统计变量
    n_ok = 0
    wer_sum = 0.0
    cer_sum = 0.0
    len_sum = 0

    # 打开输出文件
    with open(RESULT_JSONL, "w", encoding="utf-8") as f_out:
        
        # 使用 tqdm 显示进度
        for item in tqdm(data, desc="ASR Eval Offline", unit="utt"):
            ap = item["audio"]
            ref = item.get("text", "")
            b = float(item.get("begin_time", -1))
            e = float(item.get("end_time", -1))
            
            # 临时文件路径
            temp_wav_path = None
            
            try:
                # ------------------ 音频处理 ------------------
                # 判断是否需要切片
                duration = None if e < 0 else max(0, e - b)
                offset = b if b >= 0 else 0.0
                
                # 如果有特定的起止时间，我们需要手动切片并保存为临时文件
                # 因为 inference.run_stt 接收的是文件路径
                if offset > 0 or (duration is not None):
                    # 加载并切片
                    audio, sr = librosa.load(ap, offset=offset, duration=duration, sr=TARGET_SR)
                    
                    # 生成唯一文件名防止冲突
                    temp_wav_path = os.path.join(TEMP_WAV_DIR, f"{uuid.uuid4().hex}.wav")
                    
                    # 保存切片后的音频
                    sf.write(temp_wav_path, audio, sr)
                    target_infer_path = temp_wav_path
                else:
                    # 如果是全文件，直接使用原路径
                    target_infer_path = ap

                # ------------------ 模型推理 ------------------
                # 调用离线接口
                hyp = inference.run_stt(audio_path=target_infer_path)
                
                # 后处理 (参考你的示例)
                if hyp:
                    hyp = hyp.replace("<|im_end|>", "")
                else:
                    hyp = ""

            except Exception as e:
                print(f"\nError processing {ap}: {e}")
                hyp = ""
            finally:
                # ------------------ 清理临时文件 ------------------
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)

            print(f"hyp is {hyp},ref is {ref}") # 如果嫌刷屏可以注释掉

            # ------------------ 计算指标 ------------------
            ref_clean = clean_text(ref)
            hyp_clean = clean_text(hyp)

            # 计算 CER/WER (加权累加逻辑保持不变)
            # 注意：jiwer.cer/wer 返回的是错误率 (float)，乘以长度得到错误字数
            ref_len = len(list(ref_clean))
            
            cer_val = jiwer.cer(ref_clean, hyp_clean) * ref_len
            wer_val = jiwer.wer(normalize_for_wer(ref_clean), normalize_for_wer(hyp_clean)) * ref_len

            r = {
                "audio": ap,
                "begin_time": b,
                "end_time": e,
                "ref_text": ref,
                "pred_text": hyp,
                "ref_text_clean": ref_clean,
                "pred_text_clean": hyp_clean,
                "cer": cer_val,
                "wer": wer_val,
                "len": ref_len,
            }

            # 写入结果
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            f_out.flush()

            # 更新统计
            n_ok += 1
            wer_sum += r["wer"]
            cer_sum += r["cer"]
            len_sum += r["len"] if r["len"] > 0 else 1

    # ------------------ 最终结果 ------------------
    final_len = max(1, len_sum)
    print(f"\n✅ Done {n_ok} samples:")
    print(f"   Avg WER: {wer_sum/final_len*100:.2f}%")
    print(f"   Avg CER: {cer_sum/final_len*100:.2f}%")
    
    # 清理临时目录
    try:
        os.rmdir(TEMP_WAV_DIR)
    except:
        pass

if __name__ == "__main__":
    main()