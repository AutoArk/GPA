import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract default global tokens.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spark-model-dir", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    from speech_tokenizer.bicodec_tokenizer.spark_tokenizer import SparkTokenizer

    tokenizer = SparkTokenizer(str(args.spark_model_dir), device=args.device)
    result = tokenizer.tokenize([str(args.ref_audio.resolve())])
    global_tokens = result["global_tokens"][0].detach().cpu().numpy()

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, global_tokens)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
