import argparse
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForCausalLM


class WhisperEncoderWrapper(torch.nn.Module):
    def __init__(self, adapter: torch.nn.Module):
        super().__init__()
        self.whisper = adapter.whisper
        self.layer_norm = adapter.layer_norm

    def forward(self, audios: torch.Tensor) -> torch.Tensor:
        encoded = self.whisper(audios)[0]
        return self.layer_norm(encoded)


class AudioAdapterWrapper(torch.nn.Module):
    def __init__(self, adapting: torch.nn.Module):
        super().__init__()
        self.adapting = adapting

    def forward(self, merged_audio_features: torch.Tensor) -> torch.Tensor:
        return self.adapting(merged_audio_features)


def save_external_data(temp_path: Path, output_path: Path) -> None:
    data_path = output_path.with_suffix(".data")
    if output_path.exists():
        output_path.unlink()
    if data_path.exists():
        data_path.unlink()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ArkAudio whisper encoder and adapter MLP to ONNX.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--whisper-output-path", type=Path, required=True)
    parser.add_argument("--adapter-output-path", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval().to(device)

    adapter = model.audio_encoder
    whisper_wrapper = WhisperEncoderWrapper(adapter).eval()
    adapter_wrapper = AudioAdapterWrapper(adapter.adapting).eval()

    whisper_dummy = torch.randn(1, 128, 3000, dtype=dtype, device=device)
    adapter_in_dim = adapter.whisper.config.hidden_size * int(adapter.merge_factor)
    adapter_dummy = torch.randn(1, 8, adapter_in_dim, dtype=dtype, device=device)

    whisper_output = args.whisper_output_path.resolve()
    whisper_output.parent.mkdir(parents=True, exist_ok=True)
    whisper_temp = whisper_output.with_name(f"{whisper_output.stem}_temp.onnx")
    if whisper_temp.exists():
        whisper_temp.unlink()

    adapter_output = args.adapter_output_path.resolve()
    adapter_output.parent.mkdir(parents=True, exist_ok=True)
    adapter_temp = adapter_output.with_name(f"{adapter_output.stem}_temp.onnx")
    if adapter_temp.exists():
        adapter_temp.unlink()

    torch.onnx.export(
        whisper_wrapper,
        whisper_dummy,
        str(whisper_temp),
        opset_version=args.opset,
        input_names=["audios"],
        output_names=["encoded_audio_features"],
        dynamic_axes={
            "audios": {2: "mel_seq"},
            "encoded_audio_features": {1: "audio_seq"},
        },
        do_constant_folding=True,
        export_params=True,
        verbose=False,
        dynamo=False,
    )
    save_external_data(whisper_temp, whisper_output)

    torch.onnx.export(
        adapter_wrapper,
        adapter_dummy,
        str(adapter_temp),
        opset_version=args.opset,
        input_names=["merged_audio_features"],
        output_names=["audio_embeddings"],
        dynamic_axes={
            "merged_audio_features": {1: "audio_seq"},
            "audio_embeddings": {1: "audio_seq"},
        },
        do_constant_folding=True,
        export_params=True,
        verbose=False,
        dynamo=False,
    )
    save_external_data(adapter_temp, adapter_output)

    print(f"saved {whisper_output} ({args.dtype})")
    print(f"saved {adapter_output} ({args.dtype})")


if __name__ == "__main__":
    main()
