import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


BASE_PYTHONPATH = "/opt/conda/lib/python3.11/site-packages"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = f"{BASE_PYTHONPATH}:{env.get('PYTHONPATH', '')}"
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def write_manifest(model_dir: Path) -> None:
    payload = {
        "sample_rate": 16000,
        "latent_hop_length": 320,
        "global_token_offset": 151670,
        "global_tokens_shape": [1, 1, 32],
        "audio_token_id": 151663,
        "user_token_id": 151665,
        "assistant_token_id": 151668,
        "im_end_token_id": 151645,
        "pad_token_id": 151643,
        "stop_token_ids": [151645, 151643, 151665],
        "audio_encoder_default": "audio_encoder_whisper_int8.onnx",
        "audio_adapter_default": "audio_encoder_adapter_int8.onnx",
        "audio_merge_factor": 4,
        "audio_encoder_hidden_size": 1280,
        "embedding_default": "embedding_int4.onnx",
        "llm_default": "llm_kv_cpu_fp32.onnx",
        "detokenizer_default": "spark_detokenizer_int8.onnx",
    }
    (model_dir / "runtime_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ArkAudio ONNX runtime artifacts.")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--ark-model-dir", type=Path, default=Path("/data/model/autoark_audio_test"))
    parser.add_argument("--spark-model-dir", type=Path, default=Path("/data/model/SparkAudio/Spark-TTS-0___5B"))
    parser.add_argument("--ark-repo-root", type=Path, default=Path("/workspace/yumu/ark_asr"))
    parser.add_argument("--default-ref-audio", type=Path, default=Path("samples/sample.mp3"))
    parser.add_argument("--max-total-len", type=int, default=2048)
    parser.add_argument("--global-token-device", type=str, default="cuda")
    parser.add_argument("--reuse-detokenizer-dir", type=Path)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    model_dir = runtime_root / "model"
    ref_dir = model_dir / "reference"
    build_dir = runtime_root / "build"
    model_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    fp_audio_whisper = build_dir / "audio_encoder_whisper_fp32.onnx"
    fp_audio_adapter = build_dir / "audio_encoder_adapter_fp32.onnx"
    fp_embed = build_dir / "embedding_fp32.onnx"
    fp_llm = model_dir / "llm_kv_cpu_fp32.onnx"
    fp16_llm = model_dir / "llm_kv_cuda_fp16.onnx"

    run([
        sys.executable,
        str(runtime_root / "scripts" / "export_ark_audio_encoder.py"),
        "--model-dir",
        str(args.ark_model_dir),
        "--whisper-output-path",
        str(fp_audio_whisper),
        "--adapter-output-path",
        str(fp_audio_adapter),
    ])
    run([sys.executable, str(runtime_root / "scripts" / "export_ark_embedding.py"), "--model-dir", str(args.ark_model_dir), "--output-path", str(fp_embed)])
    run([
        sys.executable,
        str(runtime_root / "scripts" / "export_ark_llm_kv.py"),
        "--model-dir",
        str(args.ark_model_dir),
        "--output-path",
        str(fp_llm),
        "--max-total-len",
        str(args.max_total_len),
        "--device",
        "cpu",
        "--dtype",
        "float32",
    ])
    shutil.copy2(fp_llm.with_suffix(".json"), build_dir / "llm_kv_fp32.json")
    run([
        sys.executable,
        str(runtime_root / "scripts" / "export_ark_llm_kv.py"),
        "--model-dir",
        str(args.ark_model_dir),
        "--output-path",
        str(fp16_llm),
        "--max-total-len",
        str(args.max_total_len),
        "--device",
        "cuda",
        "--dtype",
        "float16",
    ])

    run([sys.executable, str(runtime_root / "scripts" / "quantize_dynamic_int8.py"), "--input-path", str(fp_audio_whisper), "--output-path", str(model_dir / "audio_encoder_whisper_int8.onnx")])
    run([sys.executable, str(runtime_root / "scripts" / "quantize_dynamic_int8.py"), "--input-path", str(fp_audio_adapter), "--output-path", str(model_dir / "audio_encoder_adapter_int8.onnx")])
    run([sys.executable, str(runtime_root / "scripts" / "quantize_weight_only_int4.py"), "--input-path", str(fp_embed), "--output-path", str(model_dir / "embedding_int4.onnx"), "--op-types", "Gather", "--block-size", "32"])
    run([sys.executable, str(runtime_root / "scripts" / "quantize_dynamic_int8.py"), "--input-path", str(fp_llm), "--output-path", str(model_dir / "llm_kv_cpu_fp32_int8.onnx")])
    run([sys.executable, str(runtime_root / "scripts" / "quantize_weight_only_int4.py"), "--input-path", str(fp_llm), "--output-path", str(model_dir / "llm_kv_cpu_fp32_int4.onnx"), "--op-types", "MatMul", "--block-size", "128"])

    run([sys.executable, str(runtime_root / "scripts" / "prepare_default_global_tokens.py"), "--repo-root", str(args.ark_repo_root), "--spark-model-dir", str(args.spark_model_dir), "--ref-audio", str(args.default_ref_audio), "--output-path", str(ref_dir / "default_global_tokens.npy"), "--device", str(args.global_token_device)])
    if args.reuse_detokenizer_dir is not None:
        for name in [
            "spark_detokenizer_int8.onnx",
            "spark_detokenizer_int8.onnx.data",
            "spark_detokenizer_fp16.onnx",
            "spark_detokenizer_fp16.onnx.data",
        ]:
            src = args.reuse_detokenizer_dir / name
            if src.exists():
                shutil.copy2(src, model_dir / name)
    else:
        run([sys.executable, str(runtime_root / "scripts" / "export_spark_detokenizer.py"), "--repo-root", str(args.ark_repo_root), "--spark-model-dir", str(args.spark_model_dir), "--output-dir", str(model_dir)])

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "processor_config.json",
        "preprocessor_config.json",
        "processing_arkasr.py",
    ]
    for name in tokenizer_files:
        src = args.ark_model_dir / name
        if src.exists():
            shutil.copy2(src, model_dir / name)

    write_manifest(model_dir)
    print(f"runtime ready: {runtime_root}")


if __name__ == "__main__":
    main()
