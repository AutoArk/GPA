from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class ExternalKvCacheLayer:
    is_compileable = False

    def __init__(
        self,
        cache_key: torch.Tensor,
        cache_value: torch.Tensor,
        cache_position: torch.Tensor,
        max_cache_len: int,
        is_sliding: bool,
    ) -> None:
        self.cache_key = cache_key.permute(0, 2, 1, 3).contiguous()
        self.cache_value = cache_value.permute(0, 2, 1, 3).contiguous()
        self.cache_position = cache_position
        self.max_cache_len = max_cache_len
        self.is_sliding = is_sliding
        self.is_initialized = True
        self.latest_key_delta: torch.Tensor | None = None
        self.latest_value_delta: torch.Tensor | None = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args
        del kwargs
        positions = self.cache_position.to(dtype=torch.long).view(1, 1, -1, 1)
        index = positions.expand(
            key_states.shape[0],
            key_states.shape[1],
            key_states.shape[2],
            key_states.shape[3],
        )
        updated_keys = self.cache_key.scatter(2, index, key_states)
        updated_values = self.cache_value.scatter(2, index, value_states)
        self.latest_key_delta = key_states
        self.latest_value_delta = value_states
        return updated_keys, updated_values

    def get_seq_length(self) -> torch.Tensor:
        return self.cache_position[0]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        del query_length
        return self.max_cache_len, 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class ExternalKvCache:
    is_compileable = False

    def __init__(
        self,
        cache_position: torch.Tensor,
        cache_flat: tuple[torch.Tensor, ...],
        layer_types: list[str],
        max_cache_len: int,
    ) -> None:
        self.cache_position = cache_position
        self.layers: list[ExternalKvCacheLayer] = []
        for i, layer_type in enumerate(layer_types):
            self.layers.append(
                ExternalKvCacheLayer(
                    cache_key=cache_flat[2 * i],
                    cache_value=cache_flat[2 * i + 1],
                    cache_position=cache_position,
                    max_cache_len=max_cache_len,
                    is_sliding=layer_type == "sliding_attention",
                )
            )

    @property
    def is_sliding(self) -> list[bool]:
        return [layer.is_sliding for layer in self.layers]

    def __len__(self) -> int:
        return len(self.layers)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> torch.Tensor:
        del layer_idx
        return self.cache_position[0]

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def collect_deltas(self) -> list[torch.Tensor]:
        deltas: list[torch.Tensor] = []
        for layer in self.layers:
            if layer.latest_key_delta is None or layer.latest_value_delta is None:
                raise RuntimeError("cache layer was not updated during forward")
            deltas.append(layer.latest_key_delta.permute(0, 2, 1, 3).contiguous())
            deltas.append(layer.latest_value_delta.permute(0, 2, 1, 3).contiguous())
        return deltas


class QwenNativeLlmWithKvCache(nn.Module):
    def __init__(self, model: nn.Module, max_total_len: int, emit_last_hidden_state: bool = False):
        super().__init__()
        self.core = model.model
        self.lm_head = model.lm_head
        self.max_total_len = max_total_len
        self.emit_last_hidden_state = emit_last_hidden_state

        config = self.core.config
        self.num_layers = config.num_hidden_layers
        self.layer_types = list(getattr(config, "layer_types", ["full_attention"] * self.num_layers))
        if len(self.layer_types) != self.num_layers:
            raise RuntimeError("unexpected layer_types length in model config")
        self.has_sliding_layers = "sliding_attention" in self.layer_types
        self.sliding_window = getattr(config, "sliding_window", None)

    def _build_4d_masks(
        self,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        valid_keys = attention_mask.to(dtype=torch.bool).unsqueeze(1)
        query_positions = cache_position.to(dtype=torch.long).view(1, -1, 1)
        key_positions = torch.arange(self.max_total_len, device=cache_position.device, dtype=torch.long).view(1, 1, -1)
        full_allowed = key_positions <= query_positions
        masks = {
            "full_attention": self._materialize_mask(valid_keys, full_allowed, dtype),
        }
        if self.has_sliding_layers:
            if self.sliding_window is None:
                raise RuntimeError("sliding_attention layers require config.sliding_window")
            sliding_floor = query_positions - (self.sliding_window - 1)
            sliding_allowed = full_allowed & (key_positions >= sliding_floor)
            masks["sliding_attention"] = self._materialize_mask(valid_keys, sliding_allowed, dtype)
        return masks

    @staticmethod
    def _materialize_mask(
        valid_keys: torch.Tensor,
        allowed_positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        allowed = valid_keys & allowed_positions
        zeros = torch.zeros((), dtype=dtype, device=valid_keys.device)
        neg = torch.full((), -1e4, dtype=dtype, device=valid_keys.device)
        return torch.where(allowed.unsqueeze(1), zeros, neg)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        *cache_flat: torch.Tensor,
    ):
        model_dtype = next(self.lm_head.parameters()).dtype
        outputs = self.core(
            input_ids=None,
            inputs_embeds=inputs_embeds.to(dtype=model_dtype),
            attention_mask=self._build_4d_masks(attention_mask, cache_position, model_dtype),
            past_key_values=ExternalKvCache(
                cache_position=cache_position,
                cache_flat=cache_flat,
                layer_types=self.layer_types,
                max_cache_len=self.max_total_len,
            ),
            use_cache=True,
            cache_position=cache_position,
        )
        hidden = outputs.last_hidden_state[:, -1:, :]
        logits = self.lm_head(hidden).float()

        cache = outputs.past_key_values
        deltas = cache.collect_deltas()
        if self.emit_last_hidden_state:
            return (logits, hidden.float(), *deltas)
        return (logits, *deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ArkAudio LLM with KV cache to ONNX.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-total-len", type=int, default=2048)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto")
    parser.add_argument("--emit-last-hidden-state", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.dtype == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    if device == "cpu" and dtype == torch.float16:
        raise RuntimeError("CPU export with float16 is not supported")

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval().to(device)
    model.config._attn_implementation = "eager"
    model.model.config._attn_implementation = "eager"

    wrapper = QwenNativeLlmWithKvCache(
        model,
        max_total_len=args.max_total_len,
        emit_last_hidden_state=args.emit_last_hidden_state,
    ).eval()

    config = model.model.config
    batch = 1
    seq = 8
    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden // config.num_attention_heads
    cache_dtype = next(model.parameters()).dtype

    inputs_embeds = torch.randn(batch, seq, hidden, dtype=cache_dtype, device=device)
    attention_mask = torch.zeros(batch, args.max_total_len, dtype=torch.int64, device=device)
    attention_mask[:, :seq] = 1
    cache_position = torch.arange(seq, dtype=torch.int64, device=device)
    cache_flat = []
    for _ in range(num_layers):
        cache_flat.append(
            torch.zeros(batch, args.max_total_len, num_kv_heads, head_dim, dtype=cache_dtype, device=device)
        )
        cache_flat.append(
            torch.zeros(batch, args.max_total_len, num_kv_heads, head_dim, dtype=cache_dtype, device=device)
        )

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}_temp.onnx")
    data_path = output_path.with_suffix(".data")
    if temp_path.exists():
        temp_path.unlink()
    if output_path.exists():
        output_path.unlink()
    if data_path.exists():
        data_path.unlink()

    input_names = ["inputs_embeds", "attention_mask", "cache_position"]
    output_names = ["logits"]
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch"},
        "cache_position": {0: "seq"},
        "logits": {0: "batch"},
    }
    if args.emit_last_hidden_state:
        output_names.append("last_hidden_state")
        dynamic_axes["last_hidden_state"] = {0: "batch"}
    for i in range(num_layers):
        input_names.extend([f"cache_key_{i}", f"cache_value_{i}"])
        output_names.extend([f"key_delta_{i}", f"value_delta_{i}"])
        dynamic_axes[f"cache_key_{i}"] = {0: "batch"}
        dynamic_axes[f"cache_value_{i}"] = {0: "batch"}
        dynamic_axes[f"key_delta_{i}"] = {0: "batch", 1: "seq"}
        dynamic_axes[f"value_delta_{i}"] = {0: "batch", 1: "seq"}

    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask, cache_position, *cache_flat),
        str(temp_path),
        opset_version=args.opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
        dynamo=False,
    )

    model_onnx = onnx.load(str(temp_path), load_external_data=True)
    onnx.save_model(
        model_onnx,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    temp_path.unlink()

    manifest = {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "max_total_len": args.max_total_len,
        "model_type": getattr(config, "model_type", ""),
        "sliding_window": getattr(config, "sliding_window", None),
        "layer_types": getattr(config, "layer_types", None),
        "attn_implementation": "eager",
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
