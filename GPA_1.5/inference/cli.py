import argparse
import json
from pathlib import Path

import soundfile as sf

from .asr import run_asr
from .assets import build_default_path_report, resolve_audio_tokenizer_dir, resolve_model_dir
from .model_loader import default_device, load_spark_stack, load_text_stack
from .tts import run_tts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native PyTorch inference entry for GPA v1.5 ASR and TTS.",
    )
    parser.add_argument("--model-path", "--model_path", dest="model_path", type=str, default=None)
    parser.add_argument(
        "--audio-tokenizer-path",
        "--audio_tokenizer_path",
        dest="audio_tokenizer_path",
        type=str,
        default=None,
    )
    parser.add_argument("--task", type=str, required=True, choices=["asr", "tts"])
    parser.add_argument("--print-default-paths", action="store_true")

    parser.add_argument("--asr-audio", "--asr_audio", dest="asr_audio", type=str, default=None)
    parser.add_argument("--begin-time", "--begin_time", dest="begin_time", type=float, default=-1)
    parser.add_argument("--end-time", "--end_time", dest="end_time", type=float, default=-1)
    parser.add_argument("--asr-max-new-tokens", "--asr_max_new_tokens", dest="asr_max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--asr-block-token-id-from",
        "--asr_block_token_id_from",
        dest="asr_block_token_id_from",
        type=int,
        default=151670,
    )
    parser.add_argument("--sampling-rate", "--sampling_rate", dest="sampling_rate", type=int, default=16000)
    parser.add_argument("--max-audio-seconds", "--max_audio_seconds", dest="max_audio_seconds", type=int, default=30)

    parser.add_argument("--ref-audio", "--ref_audio", dest="ref_audio", type=str, default=None)
    parser.add_argument("--tts-text", "--tts_text", dest="tts_text", type=str, default=None)
    parser.add_argument("--tts-out-wav", "--tts_out_wav", dest="tts_out_wav", type=str, default="outputs/tts_out.wav")
    parser.add_argument("--tts-stop-id", "--tts_stop_id", dest="tts_stop_id", type=int, default=151665)
    parser.add_argument("--tts-max-new-tokens", "--tts_max_new_tokens", dest="tts_max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--tts-max-semantic-tokens",
        "--tts_max_semantic_tokens",
        dest="tts_max_semantic_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument("--dump-tts-raw", "--dump_tts_raw", dest="dump_tts_raw", action="store_true")

    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument(
        "--attn-impl",
        "--attn_impl",
        dest="attn_impl",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "eager", "sdpa"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.print_default_paths:
        print(json.dumps(build_default_path_report(), indent=2))
        return

    model_path = resolve_model_dir(args.model_path)
    audio_tokenizer_path = resolve_audio_tokenizer_dir(args.audio_tokenizer_path)
    tokenizer, processor, tts_processor, model, normalized_attn_impl = load_text_stack(
        model_path=model_path,
        device=args.device,
        attn_impl=args.attn_impl,
    )

    print(f"[info] device={args.device}")
    print(f"[info] attn_impl={normalized_attn_impl}")
    print(f"[info] model_path={model_path}")
    print(f"[info] audio_tokenizer_path={audio_tokenizer_path}")

    if args.task == "asr":
        if not args.asr_audio:
            raise ValueError("--asr-audio is required for ASR")
        asr_text = run_asr(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            audio_path=args.asr_audio,
            begin_time=args.begin_time,
            end_time=args.end_time,
            sampling_rate=args.sampling_rate,
            max_audio_seconds=args.max_audio_seconds,
            max_new_tokens=args.asr_max_new_tokens,
            asr_block_token_id_from=args.asr_block_token_id_from,
            device=args.device,
        )
        print("\n========== ASR RESULT ==========")
        print(asr_text)

    if args.task == "tts":
        if not args.ref_audio or not args.tts_text:
            raise ValueError("--ref-audio and --tts-text are required for TTS")
        spark_tokenizer, spark_detokenizer = load_spark_stack(
            audio_tokenizer_path=audio_tokenizer_path,
            device=args.device,
        )
        output_path = Path(args.tts_out_wav)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        waveform, semantic_ids, raw_generation = run_tts(
            model=model,
            tts_processor=tts_processor,
            tokenizer=tokenizer,
            spark_tokenizer=spark_tokenizer,
            spark_detokenizer=spark_detokenizer,
            ref_audio_path=args.ref_audio,
            text=args.tts_text,
            tts_stop_id=args.tts_stop_id,
            max_new_tokens=args.tts_max_new_tokens,
            max_semantic_tokens=args.tts_max_semantic_tokens,
            device=args.device,
        )
        sf.write(str(output_path), waveform, args.sampling_rate)

        print("\n========== TTS RESULT ==========")
        print(f"[tts] saved wav to: {output_path}")
        print(f"[tts] semantic tokens: {len(semantic_ids)}")

        if args.dump_tts_raw:
            raw_output_path = output_path.with_suffix(".gen.txt")
            raw_output_path.write_text(raw_generation, encoding="utf-8")
            print(f"[tts] dumped raw generation to: {raw_output_path}")