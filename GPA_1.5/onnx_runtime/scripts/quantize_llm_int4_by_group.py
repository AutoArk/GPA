import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat


GROUP_PATTERNS = {
    "attention": ("/q_proj/", "/k_proj/", "/v_proj/", "/o_proj/"),
    "mlp": ("/mlp/gate_proj_", "/mlp/up_proj_", "/mlp/down_proj_"),
    "lm_head": ("/lm_head/",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--group", choices=["attention", "mlp", "lm_head"], required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--accuracy-level", type=int, default=4)
    parser.add_argument("--symmetric", action="store_true")
    args = parser.parse_args()

    model = onnx.load(str(args.input_path), load_external_data=True)
    patterns = GROUP_PATTERNS[args.group]
    include = [node.name for node in model.graph.node if node.op_type == "MatMul" and any(p in node.name for p in patterns)]
    if not include:
        raise ValueError(f"no nodes matched group={args.group}")

    config = DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=args.symmetric,
        accuracy_level=args.accuracy_level,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    quant = MatMulNBitsQuantizer(
        model=model,
        nodes_to_include=include,
        algo_config=config,
    )
    quant.process()
    quant.model.save_model_to_file(str(args.output_path), use_external_data_format=True)
    print(f"saved {args.output_path}")
    print(f"included_nodes {len(include)}")


if __name__ == "__main__":
    main()
