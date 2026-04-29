import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization helper.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--weight-type", type=str, default="quint8", choices=["qint8", "quint8"])
    args = parser.parse_args()

    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    quantize_dynamic(
        str(args.input_path.resolve()),
        str(output_path),
        weight_type=weight_type,
        per_channel=True,
        reduce_range=False,
    )
    onnx.checker.check_model(onnx.load(str(output_path)))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
