import json
from typing import Dict, Optional

from .LLMCaller import LLMCaller
from .LLMPrompts import CATEGORIES, SYSTEM_PROMPT, build_prompt
from ..VectorHandler.VectorScoring import VectorScorer


class MultiSetLLMScoring:
    def __init__(self) -> None:
        self._caller = LLMCaller()
        self._fallback = VectorScorer()

    def score_text(self, text: str) -> Dict[str, float]:
        content = text.strip()
        if not content:
            return {category: 0.0 for category in CATEGORIES}

        if self._caller.is_configured():
            prompt = build_prompt(content)
            response = self._caller.call(SYSTEM_PROMPT, prompt)
            parsed = self._parse_scores(response)
            if parsed is not None:
                return parsed

        return self._fallback.score_text(content)

    def _parse_scores(self, response: Optional[str]) -> Optional[Dict[str, float]]:
        if not response:
            return None
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return None

        result: Dict[str, float] = {}
        for category in CATEGORIES:
            value = payload.get(category)
            if isinstance(value, (int, float)):
                normalized = float(value)
                if normalized > 1.0:
                    normalized = normalized / 100.0
                result[category] = max(0.0, min(normalized, 1.0))
            else:
                result[category] = 0.0
        return result
