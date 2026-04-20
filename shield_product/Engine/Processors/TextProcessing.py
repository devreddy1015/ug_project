from typing import Dict


def build_text_context(preprocessed: Dict[str, object]) -> Dict[str, object]:
    return {
        "text": preprocessed.get("clean_text", ""),
        "length": preprocessed.get("length", 0),
    }
