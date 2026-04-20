from pathlib import Path
from typing import Dict

from .types import SemanticResult, SignalPayload


class LegacySignalExtractor:
    def extract(
        self,
        video_path: Path,
        *,
        frame_count: int,
        enable_whisper: bool,
        enable_ocr: bool,
    ) -> SignalPayload:
        from training.guard_multimodal import extract_reel_signals

        signals = extract_reel_signals(
            video_path,
            frame_count=int(frame_count),
            enable_whisper=bool(enable_whisper),
            enable_ocr=bool(enable_ocr),
        )
        return SignalPayload(
            transcript_text=signals.transcript_text,
            ocr_text=signals.ocr_text,
            caption_text=signals.caption_text,
            hashtags=list(signals.hashtags),
            frame_signals=dict(signals.frame_signals),
            metadata=dict(signals.metadata),
        )


class LegacySemanticScorer:
    def __init__(self) -> None:
        from training.guard_scoring import DualScoringEngine

        self._engine = DualScoringEngine()

    def score(self, combined_text: str) -> SemanticResult:
        result = self._engine.score(combined_text)
        return SemanticResult(
            combined_scores={key: float(value) for key, value in result.combined_scores.items()},
            llm_verdict=str(result.llm_verdict),
            vector_scores={key: float(value) for key, value in result.vector_scores.items()},
            llm_scores={key: float(value) for key, value in result.llm_scores.items()},
            engine_details={
                "used_groq": bool(result.used_groq),
                "used_embeddings": bool(result.used_embeddings),
                "used_chromadb": bool(result.used_chromadb),
                "llm_error": str(result.llm_error),
            },
        )


class LegacyEngagementProvider:
    def load(self, video_path: Path) -> Dict[str, object]:
        from training.guard_platform import load_engagement_metadata

        payload = load_engagement_metadata(video_path)
        if isinstance(payload, dict):
            cleaned: Dict[str, object] = {}
            for key, value in payload.items():
                key_name = str(key)
                if _is_number(value):
                    cleaned[key_name] = float(value)
                elif isinstance(value, str):
                    cleaned[key_name] = value
            return cleaned
        return {}


class LegacyCulturalAdapter:
    def adapt(self, category_scores: Dict[str, float], region: str) -> Dict[str, float]:
        from training.guard_platform import apply_cultural_adapter

        adapted = apply_cultural_adapter(category_scores, region)
        return {str(key): float(value) for key, value in adapted.items()}


def _is_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
