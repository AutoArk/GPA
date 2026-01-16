import os
import re
from models.bicodec_tokenizer.spark_tokenizer import SparkTokenizer
from models.glm_speech_tokenizer.speech_token_extractor import SpeechTokenExtractor
from models.glm_speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import PreTrainedTokenizer,AutoTokenizer,WhisperFeatureExtractor
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Literal, Optional, Union
from datasets import load_dataset
from torch.utils.data import DataLoader

def has_punctuation(text: str) -> bool:
    # 包含中英文符号
    pattern = r"[，。！？；：（）“”‘’、,.!?;:()\[\]{}\"']"
    return bool(re.search(pattern, text))

ALL_TASKS = ["stt", "tts-a", "vc"]


class ark_infer_processor:
    def __init__(
        self,
        glm_tokenizer: SpeechTokenExtractor,
        bicodec_tokenizer: SparkTokenizer,
        text_tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        glm_semantic_token_offset: int = 151727,
        semantic_token_offset: int = 172207,
        global_token_offset: int = 168111,
        audio_path_name: str = "audio",
        device: str = "cpu",
    ):
        self.glm_tokenizer = glm_tokenizer
        self.bicodec_tokenizer = bicodec_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.glm_semantic_token_offset = glm_semantic_token_offset
        self.semantic_token_offset = semantic_token_offset
        self.global_token_offset = global_token_offset
        self.device = device
        self.audio_path_name = audio_path_name

    def _process_example_stt(self, audio_path: str):

        ##target 音频
        with torch.no_grad():
            glm_semantic_tokens = self.glm_tokenizer.extract([audio_path])
            glm_semantic_tokens = torch.as_tensor(glm_semantic_tokens, device="cpu", dtype=torch.long)

            semantic_tokens = self.bicodec_tokenizer.tokenize([audio_path])['semantic_tokens']
        glm_semantic_tokens_list = (
            (glm_semantic_tokens + self.glm_semantic_token_offset).cpu().tolist()[0]
        )
        semantic_tokens_list = (
            (semantic_tokens + self.semantic_token_offset).cpu().tolist()[0]
        )
        input_ids = (
            self.text_tokenizer.encode("<|start_glm_token|>")
            + glm_semantic_tokens_list
            + self.text_tokenizer.encode("<|end_glm_token|>")
            + self.text_tokenizer.encode("<|start_semantic_token|>")
            + semantic_tokens_list
            + self.text_tokenizer.encode("<|end_semantic_token|>")
            + self.text_tokenizer.encode("<|start_content|>")
        )
        attention_mask = [1] * (len(input_ids))
        return input_ids, attention_mask

    def _process_example_tts_a(self, text: str, ref_audio_path: str):
        with torch.no_grad():
            global_tokens = self.bicodec_tokenizer.tokenize([ref_audio_path])['global_tokens']
        all_text = "<|start_content|>" + text + "<|end_content|>"
        global_tokens_list = (
            (global_tokens + self.global_token_offset).cpu().tolist()[0][0]
        )
        text_tokens = self.text_tokenizer(
            all_text, truncation=True, max_length=self.max_length
        )
        input_ids = (
            self.text_tokenizer.encode("<|start_global_token|>")
            + global_tokens_list
            + self.text_tokenizer.encode("<|end_global_token|>")
            + text_tokens["input_ids"]
        )
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask

    def _process_example_vc(self, audio_path: str, ref_audio_path: str):
        with torch.no_grad():
            semantic_tokens = self.bicodec_tokenizer.tokenize([audio_path])['semantic_tokens']
            new_global_tokens = self.bicodec_tokenizer.tokenize([ref_audio_path])['global_tokens']
        semantic_tokens_list = (
            (semantic_tokens + self.semantic_token_offset).cpu().tolist()[0]
        )
        new_global_tokens_list = (
            (new_global_tokens + self.global_token_offset).cpu().tolist()[0][0]
        )
        all_str = (
            "<|start_global_token|>"
            + self.text_tokenizer.decode(new_global_tokens_list)
            + "<|end_global_token|>"
            + "<|start_semantic_token|>"
            + self.text_tokenizer.decode(semantic_tokens_list)
            + "<|end_semantic_token|>"
            + "<|end_content|>"
        )

        inputs = self.text_tokenizer(all_str)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask

    def process_input(
        self,
        task: Literal["stt", "tts-a", "vc"],
        audio_path: str | None = None,
        ref_audio_path: str | None = None,
        text: str | None = None,
    ):
        """加载指定音频、特征并根据任务类型返回 token 化结果。"""

        if task == "stt":
            assert audio_path is not None
            input_ids, attention_mask = self._process_example_stt(audio_path)
        elif task == "tts-a":
            assert ref_audio_path is not None and text is not None
            input_ids, attention_mask = self._process_example_tts_a(
                text, ref_audio_path
            )
        elif task == "vc":
            assert audio_path is not None and ref_audio_path is not None
            input_ids, attention_mask = self._process_example_vc(
                audio_path, ref_audio_path
            )
        else:
            raise ValueError(
                f"Unsupported task: {task}, all supported tasks: {ALL_TASKS}"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class ark_processor:
    def __init__(self, 
                 glm_tokenizer: SpeechTokenExtractor,
                 bicodec_tokenizer: SparkTokenizer,
                 text_tokenizer:PreTrainedTokenizer,
                 max_length:int = 512,
                 glm_semantic_token_offset:int = 151727,
                 semantic_token_offset: int =172207,
                 global_token_offset: int =168111,
                 audio_path_name:str = "audio",
                 device:str ='cpu'):
        self.glm_tokenizer = glm_tokenizer 
        self.bicodec_tokenizer = bicodec_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.glm_semantic_token_offset =glm_semantic_token_offset
        self.semantic_token_offset=semantic_token_offset
        self.global_token_offset=global_token_offset
        self.device = device
        self.audio_path_name =audio_path_name

    def process_example(self, example: Dict[str, Any]):
        """
        这个函数由多个CPU进程并行执行。
        它负责加载、重采样和对单个样本进行特征提取/分词。
        """
        task = example.get("task", "stt")
        audio_path = example.get(self.audio_path_name, "")
        ref_audio_path = example.get("ref_audio", "")
        vc_audio = example.get("vc_audio", "")
        text = example.get("text", "")

        if task == "stt":
            ##target 音频
            with torch.no_grad():
                glm_semantic_tokens = self.glm_tokenizer.extract([audio_path])
                glm_semantic_tokens = torch.as_tensor(glm_semantic_tokens, device="cpu", dtype=torch.long)

                semantic_tokens = self.bicodec_tokenizer.tokenize([audio_path])['semantic_tokens']
            glm_semantic_tokens_list = (glm_semantic_tokens + self.glm_semantic_token_offset).cpu().tolist()[0]
            semantic_tokens_list = (semantic_tokens + self.semantic_token_offset).cpu().tolist()[0]
            # print(f"len of semantic is {len(semantic_tokens_list)}")
            ##对text进行token
            text_tokens = self.text_tokenizer(text, truncation=True, max_length=self.max_length)

            input_ids = self.text_tokenizer.encode("<|start_glm_token|>") + glm_semantic_tokens_list + self.text_tokenizer.encode("<|end_glm_token|>") \
                        + self.text_tokenizer.encode("<|start_semantic_token|>") + semantic_tokens_list + self.text_tokenizer.encode(
                "<|end_semantic_token|>") \
                        + self.text_tokenizer.encode("<|start_content|>") + text_tokens["input_ids"] + self.text_tokenizer.encode("<|end_content|>") \
                        + self.text_tokenizer.encode("<|im_end|>")
            attention_mask = [1] * (len(input_ids))
            labels = [-100] * (len(semantic_tokens_list) + 5 + len(glm_semantic_tokens_list)) + text_tokens["input_ids"] + self.text_tokenizer.encode(
                "<|end_content|>") + self.text_tokenizer.encode("<|im_end|>")

        elif task == "tts-a":
            with torch.no_grad():
                semantic_tokens = self.bicodec_tokenizer.tokenize([audio_path])['semantic_tokens']
                global_tokens = self.bicodec_tokenizer.tokenize([ref_audio_path])['global_tokens']
            all_text = "<|start_content|>" + text + "<|end_content|>"
            global_tokens_list = (global_tokens + self.global_token_offset).cpu().tolist()[0][0]
            text_tokens = self.text_tokenizer(all_text, truncation=True, max_length=self.max_length)
            semantic_tokens_list = (semantic_tokens + self.semantic_token_offset).cpu().tolist()[0]
            input_ids = self.text_tokenizer.encode("<|start_global_token|>") + global_tokens_list + self.text_tokenizer.encode(
                "<|end_global_token|>") + text_tokens["input_ids"] + semantic_tokens_list + self.text_tokenizer.encode("<|im_end|>")
            attention_mask = [1] * len(input_ids)
            labels = [-100] * (len(text_tokens["input_ids"]) + 2 + len(global_tokens_list)) + semantic_tokens_list + self.text_tokenizer.encode("<|im_end|>")

        elif task == "vc":
            with torch.no_grad():
                semantic_tokens = self.bicodec_tokenizer.tokenize([audio_path])['semantic_tokens']
                global_tokens = self.bicodec_tokenizer.tokenize([audio_path])['global_tokens']
                # global_tokens, semantic_tokens=self.bicodec_tokenizer.tokenize(audio_path=audio_path)
                # new_global_tokens, new_semantic_tokens=self.bicodec_tokenizer.tokenize(vc_audio,ref_audio_path)
                new_semantic_tokens = self.bicodec_tokenizer.tokenize([vc_audio])['semantic_tokens']
                new_global_tokens = self.bicodec_tokenizer.tokenize([ref_audio_path])['global_tokens']

            global_tokens_list = (global_tokens + self.global_token_offset).cpu().tolist()[0][0]
            semantic_tokens_list = (semantic_tokens + self.semantic_token_offset).cpu().tolist()[0]
            new_global_tokens_list = (new_global_tokens + self.global_token_offset).cpu().tolist()[0][0]
            new_semantic_tokens_list = (new_semantic_tokens + self.semantic_token_offset).cpu().tolist()[0]
            all_str = "<|start_global_token|>" + self.text_tokenizer.decode(new_global_tokens_list) + "<|end_global_token|>" + "<|start_semantic_token|>" + self.text_tokenizer.decode(
                semantic_tokens_list) + "<|end_semantic_token|>" + "<|end_content|>" + self.text_tokenizer.decode(new_semantic_tokens_list) + "<|im_end|>"

            ##add token and mask
            inputs = self.text_tokenizer(all_str)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            labels = [-100] * (5 + len(new_global_tokens_list) + len(semantic_tokens_list)) + new_semantic_tokens_list + self.text_tokenizer.encode("<|im_end|>")
        else:
            ##默认走stt
            with torch.no_grad():
                glm_semantic_tokens = self.glm_tokenizer.extract([audio_path])
                glm_semantic_tokens = torch.as_tensor(glm_semantic_tokens, device="cpu", dtype=torch.long)

                semantic_tokens = self.bicodec_tokenizer.tokenize([audio_path])['semantic_tokens']
            glm_semantic_tokens_list = (glm_semantic_tokens+self.glm_semantic_token_offset).cpu().tolist()[0]
            semantic_tokens_list = (semantic_tokens+self.semantic_token_offset).cpu().tolist()[0]
            # print(f"len of semantic is {len(semantic_tokens_list)}")
            ##对text进行token
            text_tokens = self.text_tokenizer(text, truncation=True, max_length=self.max_length)

            input_ids = self.text_tokenizer.encode("<|start_glm_token|>")+  glm_semantic_tokens_list + self.text_tokenizer.encode("<|end_glm_token|>") \
                    + self.text_tokenizer.encode("<|start_semantic_token|>")+  semantic_tokens_list + self.text_tokenizer.encode("<|end_semantic_token|>") \
                    + text_tokens["input_ids"] \
                    + self.text_tokenizer.encode("<|im_end|>")
            attention_mask = [1]*(len(semantic_tokens_list)+4+len(glm_semantic_tokens_list)) +text_tokens["attention_mask"] +[1]
            labels = [-100]*(len(semantic_tokens_list)+4+len(glm_semantic_tokens_list))+ text_tokens["input_ids"]+ self.text_tokenizer.encode("<|im_end|>")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            }


def create_tts_collate_fn(
    pad_token_id: int,
    processor,                     # ark_processor
    max_length: Optional[int]=None,# 传入你想要的截断上限，例如 512
    truncation_side: str = "right" # "right" 或 "left"，默认右截断
):
    """
    手动填充 + 可选截断的 collate_fn 工厂。

    参数：
        pad_token_id: 用于 input_ids 的 pad 值
        processor:    你的 ark_processor，需提供 .process_example()
        max_length:   若提供，则对每个样本在拼批前先截断到该长度
        truncation_side: "right" | "left"，决定从哪侧截断
    """
    label_pad_value = -100
    attention_mask_pad_value = 0

    def _truncate_1d(x: torch.Tensor, keep_len: int, side: str) -> torch.Tensor:
        if x.numel() <= keep_len:
            return x
        if side == "right":
            return x[:keep_len]
        elif side == "left":
            return x[-keep_len:]
        else:
            raise ValueError(f"Unsupported truncation_side: {side}")

    def _to_long_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().clone().long()
        return torch.tensor(x, dtype=torch.long)

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) 预处理（过滤空样本）
        proc = [processor.process_example(ex) for ex in examples if ex]
        proc = [d for d in proc if d and ("input_ids" in d) and ("attention_mask" in d) and ("labels" in d)]

        if len(proc) == 0:
            # 返回空批，避免 DataLoader 崩溃
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
            }

        # 2) 样本级截断（如果设置了 max_length）
        if max_length is not None:
            trimmed = []
            for ex in proc:
                ids  = _to_long_tensor(ex["input_ids"])
                mask = _to_long_tensor(ex["attention_mask"])
                labs = _to_long_tensor(ex["labels"])

                keep_len = min(max_length, ids.numel())
                ids  = _truncate_1d(ids,  keep_len, truncation_side)
                mask = _truncate_1d(mask, keep_len, truncation_side)
                labs = _truncate_1d(labs, keep_len, truncation_side)

                trimmed.append({"input_ids": ids, "attention_mask": mask, "labels": labs})
            proc = trimmed

        # 3) 计算本批最大长度（截断后再取最大）
        max_len_in_batch = max(int(len(ex["input_ids"])) for ex in proc)

        # 4) 逐样本右侧 pad 到 batch 最大长度
        padded_input_ids_list = []
        padded_attention_mask_list = []
        padded_labels_list = []

        for ex in proc:
            ids  = _to_long_tensor(ex["input_ids"])
            mask = _to_long_tensor(ex["attention_mask"])
            labs = _to_long_tensor(ex["labels"])

            need = max_len_in_batch - ids.numel()
            if need < 0:
                # 极端情况：有人为 max_length=None 时超长样本溢出
                keep_len = max_len_in_batch
                ids  = _truncate_1d(ids,  keep_len, "right")
                mask = _truncate_1d(mask, keep_len, "right")
                labs = _truncate_1d(labs, keep_len, "right")
                need = 0

            pad_dims = (0, need)
            ids  = F.pad(ids,  pad_dims, mode="constant", value=pad_token_id)
            mask = F.pad(mask, pad_dims, mode="constant", value=attention_mask_pad_value)
            labs = F.pad(labs, pad_dims, mode="constant", value=label_pad_value)

            padded_input_ids_list.append(ids)
            padded_attention_mask_list.append(mask)
            padded_labels_list.append(labs)

        # 5) 堆叠成批
        batch = {
            "input_ids": torch.stack(padded_input_ids_list, dim=0),
            "attention_mask": torch.stack(padded_attention_mask_list, dim=0),
            "labels": torch.stack(padded_labels_list, dim=0),
        }
        return batch

    return collate_fn

if __name__ == "__main__":
    device = "cuda:0"
    bicodec_audio_tokenizer_path = "/data/arki_production/model/SparkAudio/Spark-TTS-0___5B/"
    glm_speech_tokenizer_path = "/data/yumu/model/glm-4-voice-tokenizer"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(glm_speech_tokenizer_path)
    audio_model = WhisperVQEncoder.from_pretrained(glm_speech_tokenizer_path).eval().to(device)
    glm_tokenizer = SpeechTokenExtractor(model=audio_model, feature_extractor=feature_extractor, device=device)

    text_tokenizer = AutoTokenizer.from_pretrained("/data/yumu/model/ark_audio_v1_0_3_b",trust_remote_code=True)
    bicodec_tokenizer = SparkTokenizer(model_path=bicodec_audio_tokenizer_path, device=device)
    # 配置项
    DATASET_PATH = "/data/yumu/glm_asr_vllm/test/data/test_meeting.jsonl"
    MAX_LENGTH = 4096
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"将使用设备: {DEVICE}")


    # --- 2. 加载流式数据集 ---

    print(f"以流式方式加载数据集 '{DATASET_PATH}'...")
   
    
    streaming_dataset = load_dataset("json", data_files=DATASET_PATH, streaming=True)['train']
    # --- 4. 构建数据处理流水线 (Pipeline) ---

    print("正在对数据流进行shuffle，buffer_size=1000...")
    shuffled_dataset = streaming_dataset.shuffle(buffer_size=10000, seed=42)
    processor = ark_processor(
                            glm_tokenizer=glm_tokenizer,
                            bicodec_tokenizer=bicodec_tokenizer,
                            text_tokenizer=text_tokenizer,
                            device = DEVICE,
                            audio_path_name="audio")
    collate_fn = create_tts_collate_fn(text_tokenizer.pad_token_id,processor,max_length=4096)
        # 创建最终的DataLoader
    data_loader = DataLoader(
        shuffled_dataset, 
        batch_size=10, # 根据你的GPU显存和模型大小调整
        collate_fn=collate_fn,
        num_workers=0 # DataLoader的worker，负责从打乱后的流中拉取数据
    )
    print("\n--- 高性能流式 DataLoader 演示 ---")
    print("将从DataLoader中获取并展示第一个批次的数据：\n")
    first_batch = next(iter(data_loader))

    print("成功获取第一个批次！数据已在collate_fn中填充。")
    for key, value in first_batch.items():
        if value is not None:
            # print(f" - {key}: shape={value.shape}, dtype={value.dtype}")
            print(f" - {key}: shape={value.shape}, dtype={value}")