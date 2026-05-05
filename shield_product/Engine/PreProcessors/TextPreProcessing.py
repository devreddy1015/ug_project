from pathlib import Path
from typing import Dict

from ..utils import normalize_text, safe_read_text


def preprocess_text(path: Path) -> Dict[str, object]:
    raw_text = safe_read_text(path)
    cleaned_text = normalize_text(raw_text)
    return {
        "raw_text": raw_text,
        "clean_text": cleaned_text,
        "length": len(cleaned_text),
    }
