import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from infer_ark_audio_onnx import ArkAudioOnnxRuntime, build_tts_inputs


def run_once(model_path: str) -> None:
    rt = ArkAudioOnnxRuntime(Path("/workspace/yumu/ark_asr/ark_audio_onnx_runtime"))
    rt.llm = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    rt.llm_embed_dtype = np.float32 if rt.llm.get_inputs()[0].type == "tensor(float)" else np.float16
    text = "This is a debug prompt for the ONNX prefill path."
    global_tokens = np.load("/workspace/yumu/ark_asr/ark_audio_onnx_runtime/model/reference/default_global_tokens.npy")
    inputs = build_tts_inputs(rt.tts_processor, text, global_tokens)
    input_ids = inputs["input_ids"].cpu().numpy().astype(np.int64)
    embeds = rt._embed(input_ids)
    logits, caches, total_len = rt._forward_prefill(embeds)
    arr = logits[0, 0]
    print("model_path", model_path)
    print("prefill_finite", bool(np.isfinite(arr).all()))
    print("prefill_has_nan", bool(np.isnan(arr).any()))
    print("prefill_has_inf", bool(np.isinf(arr).any()))
    print("prefill_min", float(np.nanmin(arr)))
    print("prefill_max", float(np.nanmax(arr)))
    if np.isfinite(arr).any():
        print("prefill_argmax", int(np.nanargmax(arr)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    if args.model_path:
        run_once(args.model_path)
        return

    run_once("/workspace/yumu/ark_asr/ark_audio_onnx_runtime/build/llm_kv_fp32.onnx")
    run_once("/workspace/yumu/ark_asr/ark_audio_onnx_runtime/model/llm_kv_int4.onnx")


if __name__ == "__main__":
    main()
