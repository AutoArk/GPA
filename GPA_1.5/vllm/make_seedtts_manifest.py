from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_meta(root: Path, lang: str) -> list[dict[str, str]]:
    lang_dir = root / lang
    meta_path = lang_dir / "non_para_reconstruct_meta.lst"
    if not meta_path.is_file():
        meta_path = lang_dir / "meta.lst"
    rows: list[dict[str, str]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 5:
                item_id, prompt_text, prompt_rel, text, wav_rel = parts
            elif len(parts) == 4:
                item_id, prompt_text, prompt_rel, text = parts
                wav_rel = f"wavs/{item_id}.wav"
            else:
                continue
            prompt_audio = lang_dir / prompt_rel
            target_audio = lang_dir / wav_rel
            if not prompt_audio.is_file() or not target_audio.is_file():
                continue
            rows.append(
                {
                    "id": item_id,
                    "lang": lang,
                    "prompt_text": prompt_text,
                    "text": text,
                    "prompt_audio": str(prompt_audio),
                    "target_audio": str(target_audio),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 256-item SeedTTS pressure-test manifest")
    parser.add_argument("--root", default="/data3/seedtts_testset")
    parser.add_argument("--output", default="GPA_1.5/vllm/seedtts_256.jsonl")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rows = read_meta(root, "zh") + read_meta(root, "en")
    if len(rows) < args.num_samples:
        raise RuntimeError(f"Only found {len(rows)} usable rows under {root}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: args.num_samples]

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_lang: dict[str, int] = {}
    for row in selected:
        by_lang[row["lang"]] = by_lang.get(row["lang"], 0) + 1
    print(f"wrote={output} rows={len(selected)} by_lang={by_lang}")


if __name__ == "__main__":
    main()
