from pathlib import Path
from typing import Dict

from ..utils import probe_media


def preprocess_image(path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {
        "size_bytes": path.stat().st_size if path.exists() else None,
        "width": None,
        "height": None,
        "format": path.suffix.lower(),
    }
    try:
        from PIL import Image

        with Image.open(path) as image:
            info["width"], info["height"] = image.size
            info["format"] = image.format
    except Exception:
        media_info = probe_media(path)
        info["width"] = info["width"] or media_info.get("width")
        info["height"] = info["height"] or media_info.get("height")
    return info
