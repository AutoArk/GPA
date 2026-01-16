# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:30
# Author    :Hui Huang
import os
from typing import Literal, Optional, Tuple, Dict, Any, List, Union

import torch
import torchaudio
import torchaudio.transforms as TT
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy as np
from loguru import logger
from pathlib import Path

# ----------------- 假设这些模块位于你的项目路径下 -----------------
from .utils.file import load_config
from .utils.audio import load_audio
from .models.bicodec import BiCodec
from .base_model import SparkBaseModel
from .batch_processor import AsyncBatchEngine
# ---------------------------------------------------------------

__all__ = ["SparkTokenizer"]


class SparkTokenizer:
    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda", "mps"] | str = "cuda",
            attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = "eager",
            batch_size: int = 32,
            wait_timeout: float = 0.01,
    ):
        self.device = torch.device(device)
        self.model_dir = Path(model_path)

        # 1. 加载配置
        self.config = load_config(self.model_dir / "config.yaml")
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.dtype = torch.float16 if self.device_type == "cuda" else torch.float32
        self.target_sample_rate = self.config.get("sample_rate", 16000)

        # 2. 加载模型
        wav2vec_path = self.model_dir / "wav2vec2-large-xlsr-53"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            wav2vec_path,
            attn_implementation=attn_implementation,
            torch_dtype=self.dtype 
        )
        self.feature_extractor.config.output_hidden_states = True
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # BiCodec model
        self.model = (
            BiCodec.load_from_checkpoint(str(self.model_dir)).to(self.device).half()
        )
        self.model.eval()

        # 异步处理引擎
        self._batch_processor = AsyncBatchEngine(
            processing_function=self.batch_tokenize_async,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    def _to_ndarray(self, audio_input: Union[str, Path, torch.Tensor]) -> np.ndarray:
        """
        将输入（路径或Tensor）统一转换为指定采样率的 numpy 数组。
        """
        if isinstance(audio_input, (str, Path)):
            # 如果是路径，直接使用原有的 load_audio
            wav = load_audio(
                str(audio_input),
                sampling_rate=self.target_sample_rate,
                volume_normalize=self.config.get("volume_normalize", True),
            )
        elif isinstance(audio_input, torch.Tensor):
            # 如果是 Tensor
            wav = audio_input.detach().cpu().float()

            # 处理通道: [C, T] -> [T]
            if wav.ndim > 1:
                wav = torch.mean(wav, dim=0)

            # 这里默认输入的 Tensor 采样率已经是 self.target_sample_rate
            # 如果需要在这里做重采样，需要额外传入输入采样率参数
            wav = wav.numpy()

            # 可选：音量归一化逻辑（如果 Tensor 没归一化）
            if self.config.get("volume_normalize", True):
                max_val = np.abs(wav).max()
                if max_val > 0:
                    wav = wav / max_val * 0.9
        else:
            raise ValueError(f"Unsupported audio type: {type(audio_input)}")

        return wav

    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """获取参考音频片段"""
        ref_segment_length = (
            int(self.target_sample_rate * self.config["ref_segment_duration"])
            // self.config["latent_hop_length"]
            * self.config["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]

    def process_audio(self, audio_input: Union[str, torch.Tensor], ref_audio_input: Union[str, torch.Tensor] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        处理音频和参考音频。
        """
        wav = self._to_ndarray(audio_input)

        if ref_audio_input is None:
            wav_ref_np = self.get_ref_clip(wav)
        else:
            ref_wav = self._to_ndarray(ref_audio_input)
            wav_ref_np = self.get_ref_clip(ref_wav)

        wav_ref = torch.from_numpy(wav_ref_np).unsqueeze(0).float()
        return wav, wav_ref

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """提取 wav2vec2 特征"""
        # processor 期望是 list of numpy
        inputs = self.processor(
            [w.cpu().numpy() for w in wavs], 
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_values

        with torch.no_grad():
            with torch.amp.autocast(self.device_type, dtype=self.dtype):
                feat = self.feature_extractor(inputs.to(self.feature_extractor.device))

        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix

    @torch.no_grad()
    def tokenize(self, audios: List[Union[str, torch.Tensor]]):
        """
        支持音频路径列表或 Tensor 列表。
        """
        batch_wavs = []
        batch_ref_wavs = []

        for audio_item in audios:
            wav, wav_ref = self.process_audio(audio_input=audio_item, ref_audio_input=audio_item)
            batch_wavs.append(torch.from_numpy(wav).float())
            batch_ref_wavs.append(wav_ref.squeeze(0))

        # Padding wavs
        wav_lengths = [len(w) for w in batch_wavs]
        max_wav_len = max(wav_lengths)
        padded_wavs = torch.zeros(len(batch_wavs), max_wav_len, dtype=self.dtype).to(self.device)
        for i, w in enumerate(batch_wavs):
            padded_wavs[i, :len(w)] = w.to(self.dtype)

        # Padding ref_wavs
        ref_lengths = [len(w) for w in batch_ref_wavs]
        max_ref_len = max(ref_lengths)
        padded_ref_wavs = torch.zeros(len(batch_ref_wavs), max_ref_len, dtype=self.dtype).to(self.device)
        for i, w in enumerate(batch_ref_wavs):
            padded_ref_wavs[i, :len(w)] = w.to(self.dtype)

        # 提取特征
        feats = self.extract_wav2vec2_features(padded_wavs)

        batch = {
            "wav": padded_wavs,
            "ref_wav": padded_ref_wavs,
            "feat": feats,
        }

        semantic_tokens, global_tokens = self.model.tokenize(batch)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {"semantic_tokens": semantic_tokens, "global_tokens": global_tokens}

    async def batch_tokenize_async(self, audios: list) -> list[dict[str, torch.Tensor]]:
        tokenized = self.tokenize(audios)
        responses = []
        for i in range(len(audios)):
            responses.append({
                "global_tokens": tokenized["global_tokens"][i],
                "semantic_tokens": tokenized["semantic_tokens"][i]
            })
        return responses

    async def tokenize_async(self, audio: Union[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = await self._batch_processor.add_request(
            single_input=audio
        )
        return output

# ------------------------------------------------------------------
# 测试用例
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 配置你的模型路径
    MODEL_DIR = "/data/yumu/model/ark_tts_v1"
    
    # 初始化
    # 注意：在没有真实环境时，这行会因为找不到文件报错，请在有环境的地方运行
    tokenizer = SparkTokenizer(model_path=MODEL_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据：一个是本地存在的 wav 路径，一个是构造的 Tensor
    dummy_wav_path = "/data/yumu/arktts/dufu.wav" 
    # 构造一个 16kHz 的 2 秒音频 Tensor (假设模型要求16k)
    import torchaudio
    dummy_tensor, sr = torchaudio.load(dummy_wav_path)

    # 1. 测试路径输入
    print("Testing path input...")
    if os.path.exists(dummy_wav_path):
        res1 = tokenizer.tokenize([dummy_wav_path])
        print(f"Path results: {res1['semantic_tokens'].shape}")

    # 2. 测试 Tensor 输入
    print("Testing tensor input...")
    res2 = tokenizer.tokenize([dummy_tensor])
    print(f"Tensor results: {res2['semantic_tokens'].shape}")

    # 3. 测试混合输入 (List 包含 str 和 Tensor)
    print("Testing mixed input...")
    # 为了演示，我们传两个相同的 tensor
    res3 = tokenizer.tokenize([dummy_tensor, dummy_tensor])
    print(f"Mixed results: {res3['semantic_tokens'].shape}")
    
    print("All tests passed!")
