# -*- coding: utf-8 -*-
import asyncio
import os
import numpy as np
import soundfile as sf
import torch
from loguru import logger

# Assume your class is defined in ark_engine.py
from .ark_engine import ArkGPAEngine

# ================= Configure Paths (Please modify according to actual situation) =================
MODEL_PATH = "/data/yumu/model/ark_audio_v1_0_3_b/checkpoint-635000"  # Path containing LLM folder
BICODEC_PATH = "/data/yumu/model/ark_tts_v1" # Bicodec folder
GLM_PATH = "/data/yumu/arkasr/models/glm-4-voice-tokenizer" # GLM Tokenizer folder
TEXT_TOKENIZER_PATH = MODEL_PATH # Usually consistent with MODEL_PATH

# Test input file
TEST_REF_AUDIO = "/data/yumu/arktts/api_output_sf_chinese_female_neutral.wav"  # Find an existing wav file as reference
TEST_SRC_AUDIO = "/data/yumu/arktts/dufu.wav"  # Find an existing wav file as voice conversion source

# Output directory
OUTPUT_DIR = "./test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Helper Functions =================

def save_audio(audio_data, filename):
    """Save int16 format numpy array to wav file"""
    path = os.path.join(OUTPUT_DIR, filename)
    # Note: Engine output is int16, soundfile needs to specify subtype
    sf.write(path, audio_data, 16000, subtype='PCM_16')
    logger.info(f"Audio saved to: {path}")

# ================= Test Main Logic =================

async def main():
    # 1. Initialize Engine
    logger.info("Initializing engine...")
    engine = ArkGPAEngine(
        model_path=MODEL_PATH,
        bicodec_audio_tokenizer_path=BICODEC_PATH,
        glm_speech_tokenizer_path=GLM_PATH,
        text_tokenizer_path=TEXT_TOKENIZER_PATH,
        llm_device="cuda:0" if torch.cuda.is_available() else "cpu",
        backend="vllm" # Or "vllm"
    )
    logger.success("Engine initialized successfully!")

    # Prepare test parameters
    test_text = "你好，我是无界方舟的语音助理。很高兴为你服务。"
    common_kwargs = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 1024
    }

    # ------------------ 2. Test STT (Speech to Text) ------------------
    if os.path.exists(TEST_REF_AUDIO):
        logger.info(">>> Testing STT...")
        # Non-streaming
        stt_res = await engine.stt_async(audio=TEST_REF_AUDIO, **common_kwargs)
        logger.info(f"[STT Non-streaming result]: {stt_res}")

        # Streaming
        print("[STT Streaming output]: ", end="", flush=True)
        async for chunk in engine.stt_stream_async(audio=TEST_REF_AUDIO, **common_kwargs):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        logger.warning("Skipping STT test: Test audio file not found.")

    # ------------------ 3. Test TTS-A (Voice Cloning) ------------------
    if os.path.exists(TEST_REF_AUDIO):
        logger.info(">>> Testing TTS-A...")
        # Non-streaming
        audio_a = await engine.tts_a_async(text=test_text, ref_audio=TEST_REF_AUDIO, **common_kwargs)
        save_audio(audio_a, "tts_a_async.wav")

        # Streaming
        chunks = []
        async for chunk in engine.tts_a_stream_async(text=test_text, ref_audio=TEST_REF_AUDIO, **common_kwargs):
            chunks.append(chunk)
        save_audio(np.concatenate(chunks), "tts_a_stream.wav")

    # ------------------ 4. Test TTS-B (Attribute Cloning) ------------------
    if os.path.exists(TEST_REF_AUDIO):
        logger.info(">>> Testing TTS-B...")
        attr = {"gender": "female", "age": "Adult", "emotion": "Happy", "pitch": 180.0, "speed": 5.0}
        
        # Non-streaming
        audio_b = await engine.tts_b_async(text=test_text, ref_audio=TEST_REF_AUDIO, **attr, **common_kwargs)
        save_audio(audio_b, "tts_b_async.wav")

        # Streaming
        chunks = []
        async for chunk in engine.tts_b_stream_async(text=test_text, ref_audio=TEST_REF_AUDIO, **attr, **common_kwargs):
            chunks.append(chunk)
        save_audio(np.concatenate(chunks), "tts_b_stream.wav")

    # ------------------ 5. Test TTS-C (Zero Reference Attribute Driven) ------------------
    logger.info(">>> Testing TTS-C (No reference audio)...")
    attr_c = {"gender": "male", "age": "Child", "emotion": "Surprise", "pitch": 250.0, "speed": 6.0}
    
    # 非流式
    audio_c = await engine.tts_c_async(text=test_text, **attr_c, **common_kwargs)
    save_audio(audio_c, "tts_c_async.wav")

    # 流式
    chunks = []
    async for chunk in engine.tts_c_stream_async(text=test_text, **attr_c, **common_kwargs):
        chunks.append(chunk)
    save_audio(np.concatenate(chunks), "tts_c_stream.wav")

    # ------------------ 6. Test VC (Voice Conversion) ------------------
    if os.path.exists(TEST_SRC_AUDIO) and os.path.exists(TEST_REF_AUDIO):
        logger.info(">>> Testing VC (Voice Conversion)...")
        # Non-streaming
        audio_vc = await engine.vc_async(src_audio=TEST_SRC_AUDIO, ref_audio=TEST_REF_AUDIO, **common_kwargs)
        save_audio(audio_vc, "vc_async.wav")

        # 流式
        chunks = []
        async for chunk in engine.vc_stream_async(src_audio=TEST_SRC_AUDIO, ref_audio=TEST_REF_AUDIO, **common_kwargs):
            chunks.append(chunk)
        save_audio(np.concatenate(chunks), "vc_stream.wav")
    else:
        logger.warning("Skipping VC test: src or ref audio file not found.")

    logger.success("All function tests completed! Results saved in ./test_results directory.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass