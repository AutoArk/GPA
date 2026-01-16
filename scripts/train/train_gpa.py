import sys 
sys.path.append("../..")
sys.path.append("..")
from dataclasses import dataclass, field
import logging
import os
import pathlib
from typing import Dict, Optional, List
import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import AutoTokenizer ,WhisperFeatureExtractor
from transformers import Trainer, BitsAndBytesConfig
from transformers.integrations import deepspeed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from transformers import AutoModelForCausalLM
# --- Imports for Audio Data Pipeline ---
from datasets import load_dataset,Dataset

from data_utils.audio_dataset_ark_audio import ark_processor, create_tts_collate_fn
from models.bicodec_tokenizer.spark_tokenizer import SparkTokenizer
from models.glm_speech_tokenizer.speech_token_extractor import SpeechTokenExtractor
from models.glm_speech_tokenizer.modeling_whisper import WhisperVQEncoder

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    # This should now point to the base multimodal model, e.g., /data/model/ark_audio_tts
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data (.jsonl file)."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data (.jsonl file)."}
    )
    # --- New arguments for audio components ---
    glm_tokenizer_path: str = field(
        default="/data/yumu/model/glm-4-voice-tokenizer",
        metadata={"help": "Path to the GLM-4 audio tokenizer."},
    )
    bicodec_tokenizer_path: str = field(
        default="/data/arki_production/model/SparkAudio/Spark-TTS-0___5B/",
        metadata={"help": "Path to the bicodec audio tokenizer."},
    )
    # lazy_preprocess is no longer needed for the new streaming pipeline
    # lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

# ----------------------------------------------------------------------------------
# The old data processing functions (preprocess, SupervisedDataset, SftCollator,
# LazySupervisedDataset, make_supervised_data_module) are now removed, as they
# will be replaced by the components from audio_dataset.py.
# ----------------------------------------------------------------------------------

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # --- 1. Load Tokenizers and Audio Components ---
    rank0_print("Loading tokenizers and audio components...")

    # This is the main tokenizer for text, used by the Trainer and the model.
    rank0_print(f"model_args.model_name_or_path is {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        # model_max_length=training_args.model_max_length,
        # padding_side="right",
        # use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        rank0_print("Warning: pad_token_id is not set. Setting it to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # These components are used by the data processor.
    feature_extractor = WhisperFeatureExtractor.from_pretrained(data_args.glm_tokenizer_path)
    audio_model = WhisperVQEncoder.from_pretrained(data_args.glm_tokenizer_path).eval().to("cuda" if torch.cuda.is_available() else "cpu" )
    glm_tokenizer = SpeechTokenExtractor(model=audio_model, feature_extractor=feature_extractor, device= "cuda" if torch.cuda.is_available() else "cpu" )
    bicodec_tokenizer = SparkTokenizer(model_path=data_args.bicodec_tokenizer_path, device= "cuda" if torch.cuda.is_available() else "cpu" )

    # --- 2. Load Model and Prepare for Audio Tokens ---
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if training_args.use_lora and lora_args.q_lora
            else None
        ),
        attn_implementation="eager",
        # attn_implementation="flash_attention_2",
        **model_load_kwargs,
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # --- 3. Setup Audio Data Pipeline ---
    rank0_print("Setting up audio data pipeline...")

    # Instantiate the processor with all required components
    processor = ark_processor(
        glm_tokenizer=glm_tokenizer,
        bicodec_tokenizer=bicodec_tokenizer,
        text_tokenizer=tokenizer,
        audio_path_name="audio",
        max_length=training_args.model_max_length,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create the custom collate function
    data_collator = create_tts_collate_fn(
        pad_token_id=tokenizer.pad_token_id, 
        processor=processor,
        max_length=training_args.model_max_length,
    )

    # Load the streaming dataset
    rank0_print(f"Loading streaming dataset from: {data_args.data_path}")
    # train_dataset = load_dataset("json", data_files=data_args.data_path, streaming=False)['train']
    # train_dataset = Dataset.from_json(data_args.data_path)
    train_dataset = load_dataset(
        "json",
        data_files=data_args.data_path,
        streaming=False, 
        split="train",     
    )

    eval_dataset = None
    if data_args.eval_data_path:
        rank0_print(f"Loading streaming eval dataset from: {data_args.eval_data_path}")
        # eval_streaming_dataset = load_dataset("json", data_files=data_args.eval_data_path, streaming=False)['train']
        # eval_dataset = eval_streaming_dataset # No need to shuffle eval data
        eval_dataset = Dataset.from_json(data_args.eval_data_path)

    # --- 4. Start Trainer ---
    # The Trainer will automatically handle passing the correct inputs to the model.
    # Columns in the dataset not in the model's forward signature (like 'speaker_embs')
    # will be automatically ignored during the forward pass.
    # To USE speaker_embs, you must modify the model's forward() method to accept it.
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        # LoRA with DeepSpeed checkpointing has known issues, so we start fresh.
        # The original check for non-LoRA resume is kept.
        if (
            list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
            and not training_args.use_lora
        ):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
        )


if __name__ == "__main__":
    train()
