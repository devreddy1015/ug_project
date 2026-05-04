from typing import Dict

from .VectorScoring import VectorScorer


class MultiSetVectorScoring:
    def __init__(self) -> None:
        self._scorer = VectorScorer()

    def score_text(self, text: str) -> Dict[str, float]:
        return self._scorer.score_text(text)
