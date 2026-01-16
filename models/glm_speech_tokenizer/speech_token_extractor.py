import os
import sys
sys.path.append("../../..")
import io
import glob
import math
import tarfile
import torch
import torchaudio
import safetensors
from .configuration_whisper import WhisperVQConfig
from .modeling_whisper import WhisperVQEncoder, WhisperVQForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast
import asyncio
from .batch_processor import AsyncBatchEngine  # 修改为你的路径
from typing import List, Union, Tuple, Literal, Optional


class SpeechTokenExtractor:
    def __init__(
        self,
        model: WhisperVQEncoder,
        feature_extractor: WhisperFeatureExtractor,
        device: Literal["cpu", "cuda", "mps"] | str = "cuda",
        batch_size: int = 32,
        wait_timeout: float = 0.01,
    ):
        self.model = model.eval().to(device)
        self.feature_extractor = feature_extractor
        self.device = device
        self.wait_timeout = wait_timeout
        self.dtype = next(model.parameters()).dtype

        # 帧/采样 stride（用于 pad 对齐 & mask 下采样）
        self.pooling_kernel_size = getattr(model.config, "pooling_kernel_size", 1)
        self.frame_stride = (
            model.conv1.stride[0] *
            model.conv2.stride[0] *
            self.pooling_kernel_size
        )
        self.sample_stride = self.frame_stride * feature_extractor.hop_length

        # 重采样缓存（放在 device 上）
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        self._batch_processor = AsyncBatchEngine(
            processing_function=self._batch_extract_async,
            batch_size=batch_size,
            wait_timeout=wait_timeout,
        )

    # -------- I/O & 重采样：保持在 device 上 --------
    def _load_audio(self, utt: Union[str, torch.Tensor]) -> torch.Tensor:
        """读取单条音频 -> 1D float32 waveform（在 self.device 上，采样率16k）。"""
        # print(f"audio type is {type(utt)}")
        if isinstance(utt, torch.Tensor):
            # audio, sr = utt
            audio = utt.to(self.device, non_blocking=True)
        else:
            audio, sr = torchaudio.load(utt)          # CPU
            if audio.ndim > 1 and audio.size(0) > 1:  # 混单声道
                audio = audio.mean(dim=0, keepdim=True)
            audio = audio.squeeze(0).to(torch.float32).to(self.device, non_blocking=True)

        return audio  # [T] on device

    # -------- GPU 上做 feature_extractor --------
    def _extract_features_gpu(self, audios: List[torch.Tensor]) -> dict:
        """
        1) 输入统一转 CPU numpy(float32)（FE 的要求）
        2) 调用 FE，并传 device=self.device，让“输出张量”直接落在 GPU
        3) 若模型是 fp16，仅将 input_features 转 half（mask 不动）
        """
        # 1) CUDA/CPU Tensor -> CPU numpy
        np_audios = [a.detach().cpu().numpy().astype("float32") for a in audios]

       
        feats = self.feature_extractor(
            np_audios,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            device=self.device,                  # ← 用得上
            padding="longest",
            pad_to_multiple_of=self.sample_stride,
        )
    
        feats = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in feats.items()}

        # 3) 半精度对齐（只对 input_features）
        if self.dtype == torch.float16 and "input_features" in feats:
            feats["input_features"] = feats["input_features"].half()

        return feats


    def _forward(self, feats: dict) -> List[List[int]]:
        outputs = self.model(**feats)
        tokens = outputs.quantized_token_ids
        # mask 下采样对齐：conv 下采样 × pooling
        attn = feats["attention_mask"][
            :, :: self.model.conv1.stride[0] * self.model.conv2.stride[0]
        ][:, :: self.pooling_kernel_size]
        return [t[m.bool()].tolist() for t, m in zip(tokens, attn)]

    # -------- 同步批接口 --------
    def extract(self, utts: List[Union[str, torch.Tensor]]) -> List[List[int]]:
        """
        不做 30s 分片，也不做 microbatch。
        直接：加载/重采样 -> GPU 特征提取 -> 前向 -> 对齐输出。
        """
        audios = [self._load_audio(u) for u in utts]          # list[Tensor(T)] on device
        with torch.inference_mode():
            feats = self._extract_features_gpu(audios)        # on device
            return self._forward(feats)

    # -------- 异步批接口（保持你的返回协议）--------
    async def _batch_extract_async(self, utts: List[Union[str, torch.Tensor]]):
        tokens_list = await asyncio.to_thread(self.extract, utts)
        return [{"tokens": t} for t in tokens_list]

    async def extract_async(self, utt: Union[str, torch.Tensor]):
        result = await self._batch_processor.add_request(single_input=utt)
        feature = result.get("feature")
        return feature.get("tokens")