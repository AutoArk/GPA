from __future__ import annotations

from typing import Iterable

import transformers


def maybe_zero_3(param):
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except ImportError:
        return param.detach().cpu().clone()

    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            return param.data.detach().cpu().clone()

    return param.detach().cpu().clone()


def get_peft_state_maybe_zero_3(named_params: Iterable[tuple[str, object]], bias: str):
    if bias == "none":
        to_return = {name: tensor for name, tensor in named_params if "lora_" in name}
    elif bias == "all":
        to_return = {name: tensor for name, tensor in named_params if "lora_" in name or "bias" in name}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for name, tensor in named_params:
            if "lora_" in name:
                to_return[name] = tensor
                lora_bias_names.add(name.split("lora_")[0] + "bias")
            elif "bias" in name:
                maybe_lora_bias[name] = tensor
        for name, tensor in maybe_lora_bias.items():
            if name in lora_bias_names:
                to_return[name] = tensor
    else:
        raise NotImplementedError(f"Unsupported bias mode: {bias}")

    return {name: maybe_zero_3(tensor) for name, tensor in to_return.items()}


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias: str = "none") -> None:
    zero3_enabled = False
    try:
        from transformers.integrations import deepspeed

        zero3_enabled = deepspeed.is_deepspeed_zero3_enabled()
    except Exception:
        zero3_enabled = False

    if zero3_enabled:
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    elif getattr(trainer.args, "use_lora", False):
        state_dict = get_peft_state_maybe_zero_3(trainer.model.named_parameters(), bias)
    else:
        state_dict = trainer.model.state_dict()

    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)