from typing import Dict


def build_audio_context(preprocessed: Dict[str, object]) -> Dict[str, object]:
    return {
        "duration": preprocessed.get("duration"),
        "format": preprocessed.get("format"),
        "size_bytes": preprocessed.get("size_bytes"),
        "has_audio": preprocessed.get("has_audio"),
    }
