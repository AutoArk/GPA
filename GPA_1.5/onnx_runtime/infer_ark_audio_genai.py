import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og
import soundfile as sf
import torch
from transformers import AutoTokenizer

from tts_han_char_tokenizer import create_tts_han_char_processor, encode_tts_content_text


SEM_RE = re.compile(r"<\|bicodec_semantic_(\d+)\|>")
START_CONTENT_TAG = "<|start_content|>"
END_CONTENT_TAG = "<|end_content|>"


def load_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def normalize_global_tokens(global_tokens: np.ndarray) -> np.ndarray:
    global_tokens = np.asarray(global_tokens, dtype=np.int64)
    if global_tokens.ndim == 1:
        global_tokens = global_tokens[np.newaxis, np.newaxis, :]
    elif global_tokens.ndim == 2:
        global_tokens = global_tokens[:, np.newaxis, :]
    if tuple(global_tokens.shape) != (1, 1, 32):
        raise ValueError(f"unexpected global token shape: {global_tokens.shape}")
    return np.ascontiguousarray(global_tokens)


def load_manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / "runtime_manifest.json").read_text())


def build_tts_input_ids(tts_processor, text: str, global_tokens: np.ndarray) -> torch.Tensor:
    tokenizer = tts_processor.tokenizer
    global_text = "".join(f"<|bicodec_global_{int(x)}|>" for x in global_tokens.reshape(-1).tolist())
    prompt = (
        "Given the reference audio, synthesize speech for the following text in the same voice."
        f"<|start_global_token|>{global_text}<|end_global_token|>"
        f"{START_CONTENT_TAG}{text.strip()}{END_CONTENT_TAG}"
    )
    template_text = tts_processor.apply_chat_template(
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


class ArkAudioGenAiRuntime:
    def __init__(self, runtime_root: Path, genai_model_dir: Path):
        self.runtime_root = runtime_root.resolve()
        self.model_dir = self.runtime_root / "model"
        self.genai_model_dir = genai_model_dir.resolve()
        self.manifest = load_manifest(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.tts_processor = create_tts_han_char_processor(str(self.model_dir), trust_remote_code=True)
        self.model = og.Model(str(self.genai_model_dir))
        self.detok_fp16 = load_session(self.model_dir / "spark_detokenizer_fp16.onnx")
        self.detok_int8 = load_session(self.model_dir / "spark_detokenizer_int8.onnx")

        self.semantic_token_id_map = {}
        added_tokens_path = self.model_dir / "added_tokens.json"
        if added_tokens_path.exists():
            payload = json.loads(added_tokens_path.read_text())
            for token_text, token_id in payload.items():
                m = SEM_RE.fullmatch(token_text)
                if m:
                    self.semantic_token_id_map[int(token_id)] = int(m.group(1))

    def _get_detokenizer(self, precision: str) -> ort.InferenceSession:
        if precision == "fp16":
            return self.detok_fp16
        return self.detok_int8

    def run_tts(
        self,
        text: str,
        output_path: Path,
        global_token_path: Path,
        max_new_tokens: int,
        decoder_precision: str,
        temperature: float,
        repetition_penalty: float,
        random_seed: Optional[int] = None,
    ) -> Tuple[List[int], str]:
        global_tokens = normalize_global_tokens(np.load(global_token_path))
        input_ids = build_tts_input_ids(self.tts_processor, text, global_tokens).cpu().numpy().astype(np.int32)

        params = og.GeneratorParams(self.model)
        options: Dict[str, object] = {
            "do_sample": True,
            "temperature": float(temperature),
            "repetition_penalty": float(repetition_penalty),
            "max_length": int(input_ids.shape[1] + max_new_tokens),
        }
        if random_seed is not None:
            options["random_seed"] = int(random_seed)
        params.set_search_options(**options)

        generator = og.Generator(self.model, params)
        generator.append_tokens(input_ids[0])

        token_ids: List[int] = []
        stop_id = 151665
        while not generator.is_done() and len(token_ids) < max_new_tokens:
            generator.generate_next_token()
            next_token = int(generator.get_next_tokens()[0])
            if next_token == stop_id:
                break
            token_ids.append(next_token)

        gen_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        sem_ids = [self.semantic_token_id_map[t] for t in token_ids if t in self.semantic_token_id_map]
        if not sem_ids:
            sem_ids = [int(m.group(1)) for m in SEM_RE.finditer(gen_text)]
        if not sem_ids:
            raise RuntimeError("no semantic tokens produced")

        semantic_tokens = np.asarray(sem_ids, dtype=np.int64)[np.newaxis, :]
        detok = self._get_detokenizer(decoder_precision)
        audio = detok.run(None, {"semantic_tokens": semantic_tokens, "global_tokens": global_tokens.astype(np.int64)})[0]
        wav = np.asarray(audio).squeeze().astype(np.float32)
        sf.write(str(output_path), wav, int(self.manifest["sample_rate"]))
        return sem_ids, gen_text


def main() -> None:
    parser = argparse.ArgumentParser(description="ArkAudio TTS with ONNX Runtime GenAI official generate API")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--genai-model-dir", type=Path, required=True)
    parser.add_argument("--tts_text", type=str, required=True)
    parser.add_argument("--tts_out_wav", type=Path, required=True)
    parser.add_argument("--global-token-path", type=Path, required=True)
    parser.add_argument("--tts_max_new_tokens", type=int, default=256)
    parser.add_argument("--decoder-precision", type=str, choices=["int8", "fp16"], default="fp16")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition-penalty", type=float, default=1.5)
    parser.add_argument("--random-seed", type=int)
    args = parser.parse_args()

    runtime = ArkAudioGenAiRuntime(args.runtime_root, args.genai_model_dir)
    sem_ids, gen_text = runtime.run_tts(
        text=args.tts_text,
        output_path=args.tts_out_wav,
        global_token_path=args.global_token_path,
        max_new_tokens=args.tts_max_new_tokens,
        decoder_precision=args.decoder_precision,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        random_seed=args.random_seed,
    )
    print("TTS_SEMANTIC_COUNT:", len(sem_ids))
    print("TTS_GEN_TEXT:", gen_text[:500])


if __name__ == "__main__":
    main()
