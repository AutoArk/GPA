import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip())
    slug = re.sub(r"-+", "-", slug).strip("-").lower()
    return slug or f"voice-{uuid.uuid4().hex[:8]}"


class VoiceRegistry:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.resolve()
        self.registry_path = self.root_dir / "registry.json"
        self.voices_dir = self.root_dir / "items"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def _default_payload(self) -> dict[str, Any]:
        return {"version": 1, "voices": []}

    def load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            payload = self._default_payload()
            self.save(payload)
            return payload
        return json.loads(self.registry_path.read_text())

    def save(self, payload: dict[str, Any]) -> None:
        tmp_path = self.registry_path.with_suffix('.json.tmp')
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        tmp_path.replace(self.registry_path)

    def list_voices(self) -> list[dict[str, Any]]:
        payload = self.load()
        return sorted(payload["voices"], key=lambda item: (not item.get("is_default", False), item["name"]))

    def get_voice(self, name: str) -> dict[str, Any] | None:
        for voice in self.load()["voices"]:
            if voice["name"] == name:
                return voice
        return None

    def require_voice(self, name: str) -> dict[str, Any]:
        voice = self.get_voice(name)
        if voice is None:
            raise KeyError(f"Voice not found: {name}")
        return voice

    def global_token_path_for_voice(self, voice: dict[str, Any]) -> Path:
        return self.voices_dir / voice["voice_id"] / "global_tokens.npy"

    def ensure_default_from_reference(self, *, reference_token_path: Path, default_name: str = "default") -> dict[str, Any]:
        payload = self.load()
        for voice in payload["voices"]:
            if voice["name"] == default_name:
                return voice
        global_tokens = np.load(reference_token_path)
        return self.register(
            name=default_name,
            global_tokens=global_tokens,
            source_kind="bundled_reference",
            source_label=reference_token_path.name,
            overwrite=False,
            is_default=True,
            created_at=utc_now_iso(),
            fixed_voice_id="default",
        )

    def register(
        self,
        *,
        name: str,
        global_tokens: np.ndarray,
        source_kind: str,
        source_label: str,
        overwrite: bool = False,
        is_default: bool = False,
        created_at: str | None = None,
        fixed_voice_id: str | None = None,
    ) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Voice name must not be empty.")

        payload = self.load()
        existing = None
        for voice in payload["voices"]:
            if voice["name"] == clean_name:
                existing = voice
                break

        if existing is not None and not overwrite:
            raise FileExistsError(f"Voice already exists: {clean_name}")

        voice_id = fixed_voice_id or (existing["voice_id"] if existing is not None else slugify_name(clean_name))
        if existing is None and not fixed_voice_id:
            taken = {voice["voice_id"] for voice in payload["voices"]}
            if voice_id in taken:
                voice_id = f"{voice_id}-{uuid.uuid4().hex[:8]}"

        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        np.save(voice_dir / "global_tokens.npy", np.asarray(global_tokens, dtype=np.int64))

        metadata = {
            "name": clean_name,
            "voice_id": voice_id,
            "created_at": created_at or (existing["created_at"] if existing is not None else utc_now_iso()),
            "updated_at": utc_now_iso(),
            "source_kind": source_kind,
            "source_label": Path(source_label).name,
            "is_default": bool(is_default),
        }
        (voice_dir / "meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

        if existing is not None:
            payload["voices"] = [voice for voice in payload["voices"] if voice["name"] != clean_name]
        payload["voices"].append(metadata)
        payload["voices"] = sorted(payload["voices"], key=lambda item: (not item.get("is_default", False), item["name"]))
        self.save(payload)
        return metadata

    def delete_voice_dir_if_exists(self, voice_id: str) -> None:
        voice_dir = self.voices_dir / voice_id
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
