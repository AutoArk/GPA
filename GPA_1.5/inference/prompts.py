import re
from typing import Dict, List


SEM_RE = re.compile(r"<\|bicodec_semantic_(\d+)\|>")
START_CONTENT_TAG = "<|start_content|>"
END_CONTENT_TAG = "<|end_content|>"


def global_ids_to_str(global_token_ids: List[int]) -> str:
    return (
        "<|start_global_token|>"
        + "".join(f"<|bicodec_global_{int(token_id)}|>" for token_id in global_token_ids)
        + "<|end_global_token|>"
    )


def build_asr_conversation(audio_path: str, begin_time: float, end_time: float) -> List[Dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": audio_path,
                    "begin_time": begin_time,
                    "end_time": end_time,
                },
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        }
    ]


def build_tts_conversation(text: str, global_token_ids: List[int]) -> List[Dict[str, str]]:
    user_prompt = (
        "Given the reference audio, synthesize speech for the following text in the same voice."
        + global_ids_to_str(global_token_ids)
        + START_CONTENT_TAG
        + text.strip()
        + END_CONTENT_TAG
    )
    return [{"role": "user", "content": user_prompt}]


def extract_semantic_ids_from_text(generated_text: str) -> List[int]:
    return [int(match.group(1)) for match in SEM_RE.finditer(generated_text)]