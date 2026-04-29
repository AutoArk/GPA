import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from tts_han_char_tokenizer import create_tts_han_char_processor, encode_tts_content_text


START_CONTENT_TAG = "<|start_content|>"
END_CONTENT_TAG = "<|end_content|>"


def normalize_global_tokens(global_tokens: np.ndarray) -> np.ndarray:
    global_tokens = np.asarray(global_tokens, dtype=np.int64)
    if global_tokens.ndim == 1:
        global_tokens = global_tokens[np.newaxis, np.newaxis, :]
    elif global_tokens.ndim == 2:
        global_tokens = global_tokens[:, np.newaxis, :]
    if tuple(global_tokens.shape) != (1, 1, 32):
        raise ValueError(f"unexpected global token shape: {global_tokens.shape}")
    return np.ascontiguousarray(global_tokens)


def ort_dtype_to_numpy(ort_type: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if ort_type not in mapping:
        raise ValueError(f"unsupported ORT tensor type: {ort_type}")
    return mapping[ort_type]


def create_empty_cache(num_layers: int, max_total_len: int, num_kv_heads: int, head_dim: int, dtype: np.dtype):
    caches = []
    for _ in range(num_layers):
        caches.append(np.zeros((1, max_total_len, num_kv_heads, head_dim), dtype=dtype))
        caches.append(np.zeros((1, max_total_len, num_kv_heads, head_dim), dtype=dtype))
    return caches


def build_attention_mask(max_total_len: int, valid_len: int) -> np.ndarray:
    mask = np.zeros((1, max_total_len), dtype=np.int64)
    mask[:, :valid_len] = 1
    return mask


def build_input_ids(model_dir: Path, text: str, global_token_path: Path) -> torch.Tensor:
    processor = create_tts_han_char_processor(str(model_dir), trust_remote_code=True)
    tokenizer = processor.tokenizer
    global_tokens = normalize_global_tokens(np.load(global_token_path))
    global_text = "".join(f"<|bicodec_global_{int(x)}|>" for x in global_tokens.reshape(-1).tolist())
    prompt = (
        "Given the reference audio, synthesize speech for the following text in the same voice."
        f"<|start_global_token|>{global_text}<|end_global_token|>"
        f"{START_CONTENT_TAG}{text.strip()}{END_CONTENT_TAG}"
    )
    template_text = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    start_idx = template_text.index(START_CONTENT_TAG) + len(START_CONTENT_TAG)
    end_idx = template_text.index(END_CONTENT_TAG, start_idx)
    prefix_text = template_text[:start_idx]
    content_text = template_text[start_idx:end_idx]
    suffix_text = template_text[end_idx:]

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    content_ids = encode_tts_content_text(tokenizer, content_text)["input_ids"]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    return torch.tensor([prefix_ids + content_ids + suffix_ids], dtype=torch.long)


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def init_onnx_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def run_torch_step(
    model,
    input_ids: torch.Tensor,
    cache: DynamicCache | None,
    cache_position: torch.Tensor,
):
    inputs_embeds = model.model.embed_tokens(input_ids)
    outputs = model.model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    last_hidden_state = outputs.last_hidden_state[:, -1:, :].float()
    logits = model.lm_head(outputs.last_hidden_state[:, -1:, :]).float()
    return logits, last_hidden_state, outputs.past_key_values


def run_onnx_prefill(
    session: ort.InferenceSession,
    input_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    llm_meta: dict,
):
    embed_dtype = ort_dtype_to_numpy(session.get_inputs()[0].type)
    cache_input = next(inp for inp in session.get_inputs() if inp.name.startswith("cache_key_0"))
    cache_dtype = ort_dtype_to_numpy(cache_input.type)

    input_np = input_ids.cpu().numpy().astype(np.int64)
    inputs_embeds = torch.nn.functional.embedding(input_ids, embedding_weight).cpu().numpy()
    inputs_embeds = inputs_embeds.astype(embed_dtype)

    hidden_size = int(llm_meta["hidden_size"])
    num_layers = int(llm_meta["num_layers"])
    num_kv_heads = int(llm_meta["num_key_value_heads"])
    head_dim = hidden_size // int(llm_meta["num_attention_heads"])
    max_total_len = int(llm_meta["max_total_len"])
    caches = create_empty_cache(num_layers, max_total_len, num_kv_heads, head_dim, cache_dtype)

    prompt_len = input_np.shape[1]
    feeds = {
        "inputs_embeds": inputs_embeds,
        "cache_position": np.arange(prompt_len, dtype=np.int64),
    }
    input_names = {inp.name for inp in session.get_inputs()}
    if "attention_mask" in input_names:
        feeds["attention_mask"] = build_attention_mask(max_total_len, prompt_len)
    for i in range(num_layers):
        feeds[f"cache_key_{i}"] = caches[2 * i]
        feeds[f"cache_value_{i}"] = caches[2 * i + 1]

    outputs = session.run(None, feeds)
    logits = outputs[0]
    output_names = [out.name for out in session.get_outputs()]
    hidden = None
    delta_offset = 1
    if len(output_names) > 1 and output_names[1] == "last_hidden_state":
        hidden = outputs[1]
        delta_offset = 2
    deltas = outputs[delta_offset:]
    for i in range(num_layers):
        caches[2 * i][:, :prompt_len] = deltas[2 * i]
        caches[2 * i + 1][:, :prompt_len] = deltas[2 * i + 1]
    return logits, hidden, caches, prompt_len, embed_dtype


def run_onnx_decode_step(
    session: ort.InferenceSession,
    token_id: int,
    total_len: int,
    caches,
    embedding_weight: torch.Tensor,
    embed_dtype: np.dtype,
    llm_meta: dict,
):
    num_layers = int(llm_meta["num_layers"])
    token = torch.tensor([[token_id]], dtype=torch.long)
    embed = torch.nn.functional.embedding(token, embedding_weight).cpu().numpy().astype(embed_dtype)
    feeds = {
        "inputs_embeds": embed,
        "cache_position": np.array([total_len], dtype=np.int64),
    }
    input_names = {inp.name for inp in session.get_inputs()}
    if "attention_mask" in input_names:
        feeds["attention_mask"] = build_attention_mask(int(llm_meta["max_total_len"]), total_len + 1)
    for i in range(num_layers):
        feeds[f"cache_key_{i}"] = caches[2 * i]
        feeds[f"cache_value_{i}"] = caches[2 * i + 1]
    outputs = session.run(None, feeds)
    logits = outputs[0]
    output_names = [out.name for out in session.get_outputs()]
    hidden = None
    delta_offset = 1
    if len(output_names) > 1 and output_names[1] == "last_hidden_state":
        hidden = outputs[1]
        delta_offset = 2
    deltas = outputs[delta_offset:]
    for i in range(num_layers):
        caches[2 * i][:, total_len : total_len + 1] = deltas[2 * i]
        caches[2 * i + 1][:, total_len : total_len + 1] = deltas[2 * i + 1]
    return logits, hidden


def main():
    parser = argparse.ArgumentParser(description="Compare Torch and ONNX TTS decode step-by-step.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--llm-meta-json", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--global-token-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn-impl", choices=["eager", "sdpa"], default="sdpa")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch_dtype = torch_dtype_from_name(args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).eval().to(args.device)
    model.config._attn_implementation = args.attn_impl
    model.model.config._attn_implementation = args.attn_impl

    input_ids = build_input_ids(args.model_dir, args.text, args.global_token_path).to(args.device)
    llm_meta = json.loads(args.llm_meta_json.read_text())
    session = init_onnx_session(args.onnx_path)
    embedding_weight = model.model.embed_tokens.weight.detach().to(device="cpu", dtype=torch.float32)

    pt_cache = DynamicCache()
    prompt_len = input_ids.shape[1]
    pt_logits, pt_hidden, pt_cache = run_torch_step(
        model,
        input_ids=input_ids,
        cache=pt_cache,
        cache_position=torch.arange(prompt_len, device=args.device, dtype=torch.long),
    )
    onnx_logits, onnx_hidden, onnx_caches, total_len, embed_dtype = run_onnx_prefill(
        session=session,
        input_ids=input_ids.detach().to("cpu"),
        embedding_weight=embedding_weight,
        llm_meta=llm_meta,
    )

    history = []
    for step in range(args.steps):
        pt_step = pt_logits[0, 0].detach().cpu()
        onnx_step = torch.from_numpy(onnx_logits[0, 0].copy())
        pt_top = int(torch.argmax(pt_step).item())
        onnx_top = int(torch.argmax(onnx_step).item())
        max_abs_diff = float(torch.max(torch.abs(pt_step - onnx_step)).item())
        payload = {
            "step": step,
            "pt_top": pt_top,
            "onnx_top": onnx_top,
            "match": pt_top == onnx_top,
            "max_abs_diff": max_abs_diff,
        }
        if onnx_hidden is not None:
            pt_hidden_step = pt_hidden[0, 0].detach().cpu()
            onnx_hidden_step = torch.from_numpy(onnx_hidden[0, 0].copy())
            payload["hidden_max_abs_diff"] = float(torch.max(torch.abs(pt_hidden_step - onnx_hidden_step)).item())
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        )
        history.append(pt_top)

        next_token = torch.tensor([[pt_top]], device=args.device, dtype=torch.long)
        pt_logits, pt_hidden, pt_cache = run_torch_step(
            model,
            input_ids=next_token,
            cache=pt_cache,
            cache_position=torch.tensor([prompt_len + step], device=args.device, dtype=torch.long),
        )
        onnx_logits, onnx_hidden = run_onnx_decode_step(
            session=session,
            token_id=pt_top,
            total_len=total_len,
            caches=onnx_caches,
            embedding_weight=embedding_weight,
            embed_dtype=embed_dtype,
            llm_meta=llm_meta,
        )
        total_len += 1

    print("forced_tokens", history)


if __name__ == "__main__":
    main()
