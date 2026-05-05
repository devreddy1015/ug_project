from pathlib import Path
from typing import Dict

from ..utils import probe_media


def preprocess_video(path: Path) -> Dict[str, object]:
    info = probe_media(path)
    info["size_bytes"] = path.stat().st_size if path.exists() else None
    info["format"] = path.suffix.lower()
    return info
