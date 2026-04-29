import unicodedata
from typing import Dict, List, Tuple

from tokenizers import Regex, pre_tokenizers
from transformers import AutoProcessor


HAN_CHAR_PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{Han}|[^\r\n\p{L}\p{N}]?(?:(?!\p{Han})\p{L})+|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
HYPHEN_CHARS = {"-", "‑", "–", "—"}
UPPERCASE_ACRONYM_VOWELS = frozenset({"A", "E", "I", "O", "U"})


def _decode_ids(tokenizer, input_ids: List[int]) -> List[str]:
    return [tokenizer.decode([idx], skip_special_tokens=False) for idx in input_ids]


def _is_latin_letter(ch: str) -> bool:
    return ch.isalpha() and "LATIN" in unicodedata.name(ch, "")


def _is_combining_mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def _is_connector(ch: str) -> bool:
    return ch in {"'", "’", "_"}


def _is_latin_span_start(text: str, idx: int) -> bool:
    return _is_latin_letter(text[idx])


def _is_latin_span_char(text: str, idx: int) -> bool:
    ch = text[idx]
    if _is_latin_letter(ch) or _is_combining_mark(ch) or ch.isdigit():
        return True
    if _is_connector(ch):
        if idx <= 0 or idx + 1 >= len(text):
            return False
        prev_ch = text[idx - 1]
        next_ch = text[idx + 1]
        return (_is_latin_letter(prev_ch) or prev_ch.isdigit()) and (
            _is_latin_letter(next_ch) or next_ch.isdigit()
        )
    return False


def _consume_latin_span(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and _is_latin_span_char(text, idx):
        idx += 1
    return idx


def _split_content_spans(text: str) -> List[Tuple[str, str]]:
    spans: List[Tuple[str, str]] = []
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch.isspace():
            end = idx + 1
            while end < len(text) and text[end].isspace():
                end += 1
            spans.append(("space", text[idx:end]))
            idx = end
            continue
        if _is_latin_span_start(text, idx):
            end = _consume_latin_span(text, idx)
            spans.append(("latin", text[idx:end]))
            idx = end
            continue

        end = idx + 1
        while end < len(text):
            if text[end].isspace() or _is_latin_span_start(text, end):
                break
            end += 1
        spans.append(("other", text[idx:end]))
        idx = end
    return spans


def _encode_piece_with_space(tokenizer, piece: str) -> List[int]:
    return tokenizer(" " + piece, add_special_tokens=False)["input_ids"]


def _encode_fragment_with_space(tokenizer, fragment: str) -> List[int]:
    return tokenizer(" " + fragment, add_special_tokens=False)["input_ids"]


def _should_preserve_uppercase(fragment: str) -> bool:
    letters = [ch for ch in fragment if _is_latin_letter(ch)]
    return bool(letters) and all(ch.isupper() for ch in letters)


def _latin_letters(fragment: str) -> List[str]:
    return [ch for ch in fragment if _is_latin_letter(ch)]


def _is_all_caps_latin_fragment(fragment: str) -> bool:
    letters = _latin_letters(fragment)
    return bool(letters) and all(ch.isupper() for ch in letters)


def _is_acronym_like_uppercase_fragment(fragment: str) -> bool:
    letters = _latin_letters(fragment)
    if not letters or not all(ch.isupper() for ch in letters):
        return False
    if len(letters) <= 1:
        return True
    if any(ch.isdigit() for ch in fragment):
        return True
    return len(letters) <= 5 and not any(ch in UPPERCASE_ACRONYM_VOWELS for ch in letters)


def _should_force_word_mode_for_uppercase_sentence(spans: List[Tuple[str, str]]) -> bool:
    latin_fragments = [value for kind, value in spans if kind == "latin"]
    uppercase_word_like = [
        value
        for value in latin_fragments
        if _is_all_caps_latin_fragment(value) and len(_latin_letters(value)) >= 2
    ]
    if len(uppercase_word_like) < 4:
        return False
    return len(uppercase_word_like) * 10 >= len(latin_fragments) * 6


def _prepare_latin_fragment_for_tts(fragment: str, uppercase_sentence_mode: bool) -> str:
    if (
        uppercase_sentence_mode
        and _is_all_caps_latin_fragment(fragment)
        and not _is_acronym_like_uppercase_fragment(fragment)
    ):
        return fragment.lower()
    return fragment


def _normalize_latin_fragment(fragment: str) -> str:
    fragment = fragment.strip()
    if not fragment:
        return ""
    for hyphen in HYPHEN_CHARS:
        fragment = fragment.replace(hyphen, "")
    fragment = fragment.strip()
    if not fragment:
        return ""
    if _should_preserve_uppercase(fragment):
        return fragment
    return fragment.lower()


def _split_mixed_case_fragment(fragment: str) -> Tuple[str, str]:
    prefix, suffix, _ = _split_mixed_case_fragment_with_mode(fragment)
    return prefix, suffix


def _split_mixed_case_fragment_with_mode(fragment: str) -> Tuple[str, str, str]:
    saw_lower = False
    for idx, ch in enumerate(fragment):
        if not _is_latin_letter(ch):
            continue
        if ch.islower():
            saw_lower = True
            continue
        if not saw_lower:
            continue

        upper_run = 0
        probe = idx
        while probe < len(fragment) and _is_latin_letter(fragment[probe]) and fragment[probe].isupper():
            upper_run += 1
            probe += 1

        suffix_mode = "acronym" if upper_run >= 2 else "word"
        return fragment[:idx], fragment[idx:], suffix_mode
    return fragment, "", ""


def _raw_piece_units(tokenizer, fragment: str) -> List[Tuple[List[int], str]]:
    fragment = _normalize_latin_fragment(fragment)
    if not fragment:
        return []

    raw_ids = tokenizer(fragment, add_special_tokens=False)["input_ids"]
    raw_pieces = _decode_ids(tokenizer, raw_ids)
    return [([idx], piece) for idx, piece in zip(raw_ids, raw_pieces)]


def _whole_fragment_unit_with_space(tokenizer, fragment: str) -> List[Tuple[List[int], str]]:
    normalized_fragment = _normalize_latin_fragment(fragment)
    if not normalized_fragment:
        return []

    prefixed_ids = _encode_fragment_with_space(tokenizer, normalized_fragment)
    prefixed_pieces = _decode_ids(tokenizer, prefixed_ids)
    return [([idx], piece) for idx, piece in zip(prefixed_ids, prefixed_pieces)]


def _encode_uppercase_fragment_units(tokenizer, fragment: str) -> List[Tuple[List[int], str]]:
    units: List[Tuple[List[int], str]] = []
    idx = 0
    while idx < len(fragment):
        ch = fragment[idx]
        if _is_latin_letter(ch):
            prefixed_ids = _encode_piece_with_space(tokenizer, ch)
            prefixed_pieces = _decode_ids(tokenizer, prefixed_ids)
            units.extend(([token_id], piece) for token_id, piece in zip(prefixed_ids, prefixed_pieces))
            idx += 1
            continue
        if ch.isdigit():
            end = idx + 1
            while end < len(fragment) and fragment[end].isdigit():
                end += 1
            digit_ids = tokenizer(fragment[idx:end], add_special_tokens=False)["input_ids"]
            digit_pieces = _decode_ids(tokenizer, digit_ids)
            units.extend(([token_id], piece) for token_id, piece in zip(digit_ids, digit_pieces))
            idx = end
            continue

        raw_ids = tokenizer(ch, add_special_tokens=False)["input_ids"]
        raw_pieces = _decode_ids(tokenizer, raw_ids)
        units.extend(([token_id], piece) for token_id, piece in zip(raw_ids, raw_pieces))
        idx += 1
    return units


def _encode_latin_units(tokenizer, fragment: str) -> List[Tuple[List[int], str]]:
    normalized_fragment = _normalize_latin_fragment(fragment)
    if _should_preserve_uppercase(normalized_fragment):
        return _encode_uppercase_fragment_units(tokenizer, normalized_fragment)

    stripped_fragment = fragment.strip()
    for hyphen in HYPHEN_CHARS:
        stripped_fragment = stripped_fragment.replace(hyphen, "")
    stripped_fragment = stripped_fragment.strip()
    prefix_fragment, suffix_fragment, suffix_mode = _split_mixed_case_fragment_with_mode(stripped_fragment)
    if suffix_fragment:
        units: List[Tuple[List[int], str]] = []
        units.extend(_whole_fragment_unit_with_space(tokenizer, prefix_fragment))
        if suffix_mode == "acronym":
            for ch in suffix_fragment:
                prefixed_ids = _encode_piece_with_space(tokenizer, ch)
                prefixed_pieces = _decode_ids(tokenizer, prefixed_ids)
                units.extend(([idx], piece) for idx, piece in zip(prefixed_ids, prefixed_pieces))
        else:
            units.extend(_encode_latin_units(tokenizer, suffix_fragment))
        return units

    whole_fragment_unit = _whole_fragment_unit_with_space(tokenizer, fragment)
    if whole_fragment_unit:
        return whole_fragment_unit

    units: List[Tuple[List[int], str]] = []
    for raw_ids, raw_piece in _raw_piece_units(tokenizer, fragment):
        piece_text = raw_piece.strip()
        if not piece_text:
            continue

        prefixed_ids = _encode_piece_with_space(tokenizer, piece_text)
        prefixed_pieces = _decode_ids(tokenizer, prefixed_ids)
        if len(prefixed_pieces) == 1 and prefixed_pieces[0] == " " + piece_text:
            units.append((prefixed_ids, prefixed_pieces[0]))
        else:
            units.append((raw_ids, raw_piece))
    return units


def _neighbor_kind(spans: List[Tuple[str, str]], idx: int, step: int) -> str:
    probe = idx + step
    while 0 <= probe < len(spans):
        kind, _ = spans[probe]
        if kind != "space":
            return kind
        probe += step
    return ""


def _is_droppable_hyphen_span(spans: List[Tuple[str, str]], idx: int) -> bool:
    kind, value = spans[idx]
    if kind != "other":
        return False
    stripped = value.strip()
    if not stripped or any(ch not in HYPHEN_CHARS for ch in stripped):
        return False
    prev_kind = _neighbor_kind(spans, idx, -1)
    next_kind = _neighbor_kind(spans, idx, 1)
    return prev_kind == "latin" or next_kind == "latin"


def normalize_tts_content_text(tokenizer, text: str) -> str:
    parts: List[str] = []
    spans = _split_content_spans(text)
    uppercase_sentence_mode = _should_force_word_mode_for_uppercase_sentence(spans)
    for idx, (kind, value) in enumerate(spans):
        if kind == "latin":
            value = _prepare_latin_fragment_for_tts(value, uppercase_sentence_mode)
            for _, piece in _encode_latin_units(tokenizer, value):
                parts.append(piece)
            continue

        if _is_droppable_hyphen_span(spans, idx):
            continue

        if kind == "space":
            prev_kind = _neighbor_kind(spans, idx, -1)
            next_kind = _neighbor_kind(spans, idx, 1)
            if prev_kind == "latin" or next_kind == "latin":
                continue
        parts.append(value)
    return "".join(parts)


def encode_tts_content_text(tokenizer, text: str) -> Dict[str, List[int] | str]:
    input_ids: List[int] = []
    pieces: List[str] = []
    spans = _split_content_spans(text)
    uppercase_sentence_mode = _should_force_word_mode_for_uppercase_sentence(spans)

    for idx, (kind, value) in enumerate(spans):
        if kind == "latin":
            value = _prepare_latin_fragment_for_tts(value, uppercase_sentence_mode)
            for unit_ids, unit_piece in _encode_latin_units(tokenizer, value):
                input_ids.extend(unit_ids)
                pieces.append(unit_piece)
            continue

        if _is_droppable_hyphen_span(spans, idx):
            continue

        if kind == "space":
            prev_kind = _neighbor_kind(spans, idx, -1)
            next_kind = _neighbor_kind(spans, idx, 1)
            if prev_kind == "latin" or next_kind == "latin":
                continue

        span_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        input_ids.extend(span_ids)
        pieces.extend(_decode_ids(tokenizer, span_ids))

    return {
        "input_ids": input_ids,
        "tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "pieces": pieces,
        "full": tokenizer.decode(input_ids, skip_special_tokens=False),
        "normalized_text": normalize_tts_content_text(tokenizer, text),
    }


def create_tts_han_char_processor(model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not expose a tokenizer")

    backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        raise TypeError("TTS Han-char mode requires a fast tokenizer backend")

    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(HAN_CHAR_PRETOKENIZER_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    return processor


def describe_tokenization(tokenizer, text: str) -> Dict[str, List[int]]:
    encoded = tokenizer(text, add_special_tokens=False)
    return {
        "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
        "input_ids": encoded["input_ids"],
    }
