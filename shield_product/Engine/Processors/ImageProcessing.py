from typing import Dict


def build_image_context(preprocessed: Dict[str, object]) -> Dict[str, object]:
    return {
        "width": preprocessed.get("width"),
        "height": preprocessed.get("height"),
        "format": preprocessed.get("format"),
        "size_bytes": preprocessed.get("size_bytes"),
    }
