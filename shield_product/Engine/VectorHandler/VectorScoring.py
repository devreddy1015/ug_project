import json
from pathlib import Path
from typing import Dict

from ..utils import tokenize


DEFAULT_BAGS_PATH = Path(__file__).parent / "CategoryBags" / "category_terms.json"


class VectorScorer:
    def __init__(self, bags_path: Path = DEFAULT_BAGS_PATH) -> None:
        self._bags_path = bags_path
        self._categories = self._load_bags()

    def _load_bags(self) -> Dict[str, set]:
        if not self._bags_path.exists():
            return {}
        payload = json.loads(self._bags_path.read_text(encoding="utf-8"))
        return {key: set(value) for key, value in payload.items()}

    def score_text(self, text: str) -> Dict[str, float]:
        tokens = set(tokenize(text))
        scores: Dict[str, float] = {}
        for category, keywords in self._categories.items():
            if not keywords:
                scores[category] = 0.0
                continue
            intersection = tokens.intersection(keywords)
            union = tokens.union(keywords)
            scores[category] = round(len(intersection) / max(len(union), 1), 4)
        return scores
