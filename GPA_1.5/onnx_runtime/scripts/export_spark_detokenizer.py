import argparse
import sys
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


def export_model(model: torch.nn.Module, output_path: Path, dtype: torch.dtype) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.with_suffix(".onnx.data")
    if output_path.exists():
        output_path.unlink()
    if data_path.exists():
        data_path.unlink()

    model = model.to(dtype=dtype).eval()
    semantic_tokens = torch.randint(0, 8192, (1, 128), dtype=torch.long)
    global_tokens = torch.randint(0, 1024, (1, 1, 32), dtype=torch.long)
    torch.onnx.export(
        model,
        (semantic_tokens, global_tokens),
        str(output_path),
        opset_version=17,
        input_names=["semantic_tokens", "global_tokens"],
        output_names=["audio"],
        dynamic_axes={
            "semantic_tokens": {0: "batch", 1: "semantic_seq"},
            "global_tokens": {0: "batch"},
            "audio": {0: "batch", 2: "audio_seq"},
        },
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    model_onnx = onnx.load(str(output_path), load_external_data=False)
    onnx.save_model(
        model_onnx,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Spark detokenizer to FP16 and INT8 ONNX.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spark-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    from speech_tokenizer.bicodec_tokenizer.spark_detokenizer import SparkDeTokenizerModel

    base_model = SparkDeTokenizerModel.from_pretrained(str(args.spark_model_dir / "BiCodec")).cpu().eval()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_temp = output_dir / "spark_detokenizer_fp32_tmp.onnx"
    fp16_path = output_dir / "spark_detokenizer_fp16.onnx"
    int8_path = output_dir / "spark_detokenizer_int8.onnx"

    export_model(base_model.float(), fp32_temp, torch.float32)
    export_model(base_model.half(), fp16_path, torch.float16)
    if int8_path.exists():
        int8_path.unlink()
    quantize_dynamic(
        str(fp32_temp),
        str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    tmp_data = fp32_temp.with_suffix(".onnx.data")
    if fp32_temp.exists():
        fp32_temp.unlink()
    if tmp_data.exists():
        tmp_data.unlink()
    print(f"saved {fp16_path}")
    print(f"saved {int8_path}")


if __name__ == "__main__":
    main()
