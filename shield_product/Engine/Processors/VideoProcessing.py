from typing import Dict


def build_video_context(preprocessed: Dict[str, object]) -> Dict[str, object]:
    return {
        "duration": preprocessed.get("duration"),
        "width": preprocessed.get("width"),
        "height": preprocessed.get("height"),
        "format": preprocessed.get("format"),
        "size_bytes": preprocessed.get("size_bytes"),
        "has_audio": preprocessed.get("has_audio"),
        "has_video": preprocessed.get("has_video"),
    }
