import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM


TOKENIZER_FILES = [
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def build_qwen2_config(src_dir: Path) -> Qwen2Config:
    raw = json.loads((src_dir / "config.json").read_text())
    keep = {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling",
        "sliding_window",
        "use_sliding_window",
        "max_window_layers",
        "attention_dropout",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
        "use_cache",
    }
    cfg = {k: raw[k] for k in keep if k in raw}
    cfg["model_type"] = "qwen2"
    cfg["architectures"] = ["Qwen2ForCausalLM"]
    cfg["use_cache"] = True
    return Qwen2Config(**cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a text-only Qwen2 HF dir from ArkASR weights.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    src_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        trust_remote_code=True,
        torch_dtype=getattr(torch, args.dtype),
        device_map="cpu",
    ).eval()

    cfg = build_qwen2_config(src_dir)
    dst_model = Qwen2ForCausalLM(cfg).eval()

    filtered = {}
    for key, value in src_model.state_dict().items():
        if key.startswith("model.") or key.startswith("lm_head."):
            filtered[key] = value

    missing, unexpected = dst_model.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch when preparing qwen2 text-only model; missing={missing}, unexpected={unexpected}"
        )

    dst_model.to(dtype=getattr(torch, args.dtype))
    dst_model.save_pretrained(str(out_dir), safe_serialization=True)

    for name in TOKENIZER_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    print(f"saved text-only qwen2 hf dir to {out_dir}")


if __name__ == "__main__":
    main()
