import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat


def parse_op_types(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one op type is required")
    return values


def parse_nodes(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight-only INT4 quantization with ONNX Runtime MatMulNBits/GatherBlockQuantized.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--op-types", type=str, default="MatMul")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--accuracy-level", type=int, default=4)
    parser.add_argument("--nodes-to-exclude", type=str)
    parser.add_argument("--nodes-to-include", type=str)
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    quant_config = DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=args.symmetric,
        accuracy_level=args.accuracy_level,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=parse_op_types(args.op_types),
        bits=4,
    )
    quant = MatMulNBitsQuantizer(
        model=str(input_path),
        nodes_to_exclude=parse_nodes(args.nodes_to_exclude),
        nodes_to_include=parse_nodes(args.nodes_to_include),
        algo_config=quant_config,
    )
    quant.process()
    quant.model.save_model_to_file(str(output_path), use_external_data_format=True)
    onnx.checker.check_model(onnx.load(str(output_path), load_external_data=True), full_check=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
