import argparse
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForCausalLM


class EmbeddingWrapper(torch.nn.Module):
    def __init__(self, embed_tokens: torch.nn.Module):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ArkAudio token embedding to ONNX.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval().to(device)

    wrapper = EmbeddingWrapper(model.model.embed_tokens).eval()
    dummy = torch.randint(0, model.config.vocab_size, (1, 16), dtype=torch.long, device=device)

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}_temp.onnx")
    data_path = output_path.with_suffix(".data")
    if temp_path.exists():
        temp_path.unlink()
    if output_path.exists():
        output_path.unlink()
    if data_path.exists():
        data_path.unlink()

    torch.onnx.export(
        wrapper,
        dummy,
        str(temp_path),
        opset_version=args.opset,
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "inputs_embeds": {0: "batch", 1: "seq"},
        },
        do_constant_folding=True,
        export_params=True,
        verbose=False,
        dynamo=False,
    )

    model_onnx = onnx.load(str(temp_path), load_external_data=True)
    onnx.save_model(
        model_onnx,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    temp_path.unlink()
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
