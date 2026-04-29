import argparse
import subprocess
from pathlib import Path


CASES = [
    {"name": "bs128_acc4", "block_size": 128, "accuracy_level": 4, "symmetric": False},
    {"name": "bs128_acc1", "block_size": 128, "accuracy_level": 1, "symmetric": False},
    {"name": "bs64_acc4", "block_size": 64, "accuracy_level": 4, "symmetric": False},
    {"name": "bs32_acc4", "block_size": 32, "accuracy_level": 4, "symmetric": False},
    {"name": "bs128_acc4_sym", "block_size": 128, "accuracy_level": 4, "symmetric": True},
]


def run(cmd: list[str], cwd: Path) -> str:
    out = subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True)
    return out.stdout + out.stderr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    quant_script = runtime_root / "scripts" / "quantize_weight_only_int4.py"
    debug_script = runtime_root / "scripts" / "debug_tts_prefill.py"
    fp32_model = runtime_root / "build" / "llm_kv_fp32.onnx"
    model_dir = runtime_root / "model"

    for case in CASES:
        output_path = model_dir / f"llm_kv_{case['name']}.onnx"
        cmd = [
            "python",
            str(quant_script),
            "--input-path",
            str(fp32_model),
            "--output-path",
            str(output_path),
            "--op-types",
            "MatMul",
            "--block-size",
            str(case["block_size"]),
            "--accuracy-level",
            str(case["accuracy_level"]),
        ]
        if case["symmetric"]:
            cmd.append("--symmetric")
        print("quantizing", case["name"])
        print(run(cmd, runtime_root))
        print("debugging", case["name"])
        dbg = run(["python", str(debug_script), "--model-path", str(output_path)], runtime_root)
        print(dbg)


if __name__ == "__main__":
    main()
