from __future__ import annotations

import logging
import os
import pathlib


def build_parser():
    import transformers

    from .arguments import DataArguments, LoraArguments, ModelArguments, TrainingArguments

    return transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))


def _resolve_attn_impl(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "flash_attention_2"
    if torch.backends.mps.is_available():
        return "sdpa"
    return "eager"


def _load_tokenizer(model_path, rank0_print):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        rank0_print(f"[warn] Fast tokenizer load failed: {exc}")
        rank0_print("[warn] Falling back to use_fast=False. Check whether tokenizer.json is fully materialized.")
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


def _freeze_audio_encoder(model) -> None:
    if hasattr(model, "audio_encoder"):
        if hasattr(model.audio_encoder, "whisper"):
            for param in model.audio_encoder.whisper.parameters():
                param.requires_grad = False
        if hasattr(model.audio_encoder, "adapting"):
            for param in model.audio_encoder.adapting.parameters():
                param.requires_grad = False
        if hasattr(model.audio_encoder, "layer_norm"):
            for param in model.audio_encoder.layer_norm.parameters():
                param.requires_grad = False
        return

    for name, param in model.named_parameters():
        if "audio_encoder.whisper" in name or "audio_encoder.adapting" in name or "audio_encoder.layer_norm" in name:
            param.requires_grad = False


def _print_trainable(model) -> None:
    from .data import rank0_print

    trainable = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(numel for _, numel in trainable)
    percent = 0.0 if total_params == 0 else trainable_params / total_params * 100
    rank0_print(f"[trainable] {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M ({percent:.4f}%)")


def _prepare_lora_if_needed(model, training_args, lora_args):
    if not training_args.use_lora:
        return model

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("peft is required when --use_lora is enabled.") from exc

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
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _load_quantization_config(training_args, lora_args, compute_dtype):
    if not training_args.use_lora or not lora_args.q_lora:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("bitsandbytes support is required when --q_lora is enabled.") from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def train(argv: list[str] | None = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Trainer

    from spark_tokenizer_runtime.spark_tokenizer import SparkTokenizer
    from tts_han_char_tokenizer import create_tts_han_char_processor, describe_tokenization

    from .assets import resolve_audio_tokenizer_dir, resolve_model_dir
    from .data import ArkasrDataCollator, load_json_dataset_rank0_build_all_load, rank0_print, resolve_dataset_media_paths
    from .save_utils import safe_save_model_for_hf_trainer

    parser = build_parser()
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses(args=argv)

    try:
        from accelerate.utils import DistributedType
        from transformers.trainer_pt_utils import AcceleratorConfig
    except ImportError as exc:
        raise ImportError(
            "Training requires accelerate. Install GPA-v1.5/requirements.train.txt first."
        ) from exc

    if getattr(training_args, "deepspeed", None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        try:
            training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
        except Exception:
            logging.warning("Could not mark the run as DeepSpeed under single-process launch; continuing.")

    training_args.accelerator_config = AcceleratorConfig(dispatch_batches=False, split_batches=False)
    if not data_args.data_path:
        raise ValueError("--data_path is required.")

    model_path = resolve_model_dir(model_args.model_name_or_path)
    audio_tokenizer_path = resolve_audio_tokenizer_dir(model_args.audio_tokenizer_path)
    data_args.hf_cache_dir = os.path.expanduser(data_args.hf_cache_dir)
    os.makedirs(data_args.hf_cache_dir, exist_ok=True)
    os.makedirs(training_args.output_dir, exist_ok=True)

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    normalized_attn_impl = _resolve_attn_impl(model_args.attn_impl)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None

    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if getattr(training_args, "fsdp", None) or getattr(training_args, "deepspeed", None):
            logging.warning("FSDP or DeepSpeed ZeRO-3 is incompatible with QLoRA.")

    model_load_kwargs = {"low_cpu_mem_usage": not bool(getattr(training_args, "deepspeed", None))}

    rank0_print("Loading tokenizer and processors...")
    rank0_print(f"[info] model_path={model_path}")
    rank0_print(f"[info] audio_tokenizer_path={audio_tokenizer_path}")
    rank0_print(f"[info] attn_impl={normalized_attn_impl}")

    tokenizer = _load_tokenizer(model_path, rank0_print)
    if tokenizer.pad_token_id is None:
        rank0_print("[warn] pad_token_id is not set; using eos_token_id as pad.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tts_processor = create_tts_han_char_processor(model_path, trust_remote_code=True)

    for probe_text in ["Committee report", "Hello world"]:
        base_desc = describe_tokenization(processor.tokenizer, probe_text)
        tts_desc = describe_tokenization(tts_processor.tokenizer, probe_text)
        rank0_print(
            f"[tts-tokenizer] text={probe_text} base_tokens={base_desc['tokens']} tts_tokens={tts_desc['tokens']}"
        )

    rank0_print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=_load_quantization_config(training_args, lora_args, compute_dtype),
        attn_implementation=normalized_attn_impl,
        **model_load_kwargs,
    )
    model.config.use_cache = False

    _freeze_audio_encoder(model)
    _print_trainable(model)
    model = _prepare_lora_if_needed(model, training_args, lora_args)

    spark_device = "cuda" if torch.cuda.is_available() else "cpu"
    spark_tokenizer = SparkTokenizer(str(audio_tokenizer_path), device=spark_device)
    data_collator = ArkasrDataCollator(
        processor=processor,
        tts_processor=tts_processor,
        max_length=training_args.tts_max_semantic_tokens,
        max_audio_seconds=training_args.max_audio_seconds,
        sampling_rate=training_args.sampling_rate,
        spark_tokenizer=spark_tokenizer,
        tts_missing_token_policy=training_args.tts_missing_token_policy,
    )

    rank0_print("Loading map-style datasets...")
    train_dataset = load_json_dataset_rank0_build_all_load(data_args.data_path, data_args.hf_cache_dir)
    train_dataset = resolve_dataset_media_paths(train_dataset, data_args.data_path)
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = load_json_dataset_rank0_build_all_load(data_args.eval_data_path, data_args.hf_cache_dir)
        eval_dataset = resolve_dataset_media_paths(eval_dataset, data_args.eval_data_path)

    rank0_print(f"train_dataset size = {len(train_dataset)}")
    if eval_dataset is not None:
        rank0_print(f"eval_dataset size = {len(eval_dataset)}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if not training_args.do_train:
        rank0_print("[info] --do_train is not set. Exiting after initialization.")
        return

    existing_checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if existing_checkpoints and not training_args.use_lora:
        rank0_print("[info] Found checkpoints in output_dir, resuming from checkpoint.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias,
    )
