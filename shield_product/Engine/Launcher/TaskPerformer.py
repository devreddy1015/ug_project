from dataclasses import asdict
from datetime import datetime, timezone
import importlib
from pathlib import Path
from typing import Dict, Optional

from ..DataObjects.Base import BaseDataObject
from ..LLMHandler.MultiSetLLMScoring import MultiSetLLMScoring
from ..LLMHandler.LLMPrompts import CATEGORIES
from ..VectorHandler.MultiSetVectorScoring import MultiSetVectorScoring
from ..PreProcessors.TextPreProcessing import preprocess_text
from ..PreProcessors.ImagePreProcessing import preprocess_image
from ..PreProcessors.AudioPreProcessing import preprocess_audio
from ..PreProcessors.VideoPreprocessing import preprocess_video
from ..Processors.TextProcessing import build_text_context
from ..Processors.ImageProcessing import build_image_context
from ..Processors.AudioProcessing import build_audio_context
from ..Processors.VideoProcessing import build_video_context
from ..utils import normalize_text
from Runner.Monitor.LogMonitor import LogMonitor


class TaskPerformer:
    def __init__(
        self,
        monitor: LogMonitor,
        enable_transcription: bool = False,
        enable_ocr: bool = False,
    ) -> None:
        self._monitor = monitor
        self._llm = MultiSetLLMScoring()
        self._vector = MultiSetVectorScoring()
        self._enable_transcription = enable_transcription
        self._enable_ocr = enable_ocr

    def process_item(self, item: BaseDataObject) -> Dict[str, object]:
        preprocessed, context = self._prepare_context(item)
        text_context = context.get("text", "")
        if item.modality in {"audio", "video"} and self._enable_transcription:
            text_context = self._extract_transcript(item.path) or text_context
        if item.modality == "image" and self._enable_ocr:
            text_context = self._extract_ocr(item.path) or text_context
        text_context = normalize_text(text_context)

        llm_scores = self._llm.score_text(text_context)
        vector_scores = self._vector.score_text(text_context)
        combined_scores = self._combine_scores(llm_scores, vector_scores)
        harm_score = max(combined_scores.values()) if combined_scores else 0.0
        llm_scores_pct = self._to_percent(llm_scores)
        vector_scores_pct = self._to_percent(vector_scores)
        combined_scores_pct = self._to_percent(combined_scores)
        harm_score_pct = round(harm_score * 100.0, 4)
        policy = self._policy_summary(combined_scores_pct)

        result = {
            "file_path": str(item.path),
            "modality": item.modality,
            "label": item.label,
            "metadata": asdict(item).get("metadata", {}),
            "preprocessed": preprocessed,
            "context": context,
            "llm_scores": llm_scores,
            "llm_scores_pct": llm_scores_pct,
            "vector_scores": vector_scores,
            "vector_scores_pct": vector_scores_pct,
            "combined_scores": combined_scores,
            "combined_scores_pct": combined_scores_pct,
            "harm_score": harm_score,
            "harm_score_pct": harm_score_pct,
            "view_recommendation": policy["view_recommendation"],
            "age_preference": policy["age_preference"],
            "warning_notes": policy["warning_notes"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._monitor.publish(result)
        return result

    def _prepare_context(self, item: BaseDataObject) -> tuple[Dict[str, object], Dict[str, object]]:
        if item.modality == "text":
            preprocessed = preprocess_text(item.path)
            context = build_text_context(preprocessed)
        elif item.modality == "image":
            preprocessed = preprocess_image(item.path)
            context = build_image_context(preprocessed)
        elif item.modality == "audio":
            preprocessed = preprocess_audio(item.path)
            context = build_audio_context(preprocessed)
        else:
            preprocessed = preprocess_video(item.path)
            context = build_video_context(preprocessed)
        return preprocessed, context

    def _combine_scores(
        self, llm_scores: Dict[str, float], vector_scores: Dict[str, float]
    ) -> Dict[str, float]:
        categories = CATEGORIES
        combined: Dict[str, float] = {}
        for category in categories:
            llm_value = llm_scores.get(category, 0.0)
            vector_value = vector_scores.get(category, 0.0)
            combined[category] = round(0.6 * llm_value + 0.4 * vector_value, 4)
        return combined

    def _to_percent(self, scores: Dict[str, float]) -> Dict[str, float]:
        return {category: round(value * 100.0, 4) for category, value in scores.items()}

    def _policy_summary(self, scores_pct: Dict[str, float]) -> Dict[str, object]:
        if not scores_pct:
            return {
                "view_recommendation": "safe",
                "age_preference": "all_ages",
                "warning_notes": [],
            }

        severe = any(score >= 70.0 for score in scores_pct.values())
        high = any(score >= 40.0 for score in scores_pct.values())

        if severe:
            view_recommendation = "do_not_show"
        elif high:
            view_recommendation = "caution"
        else:
            view_recommendation = "safe"

        age_preference = "all_ages"
        if self._max_score(scores_pct, {
            "sexually_explicit_material",
            "child_abuse_and_exploitation",
            "violence_and_terrorism",
            "self_harm_and_suicide",
        }) >= 40.0:
            age_preference = "18_plus"
        elif self._max_score(scores_pct, {
            "harassment",
            "hate_speeches",
            "cyberbullying",
            "homophobic_content",
            "transphobic_content",
            "racial_slurs",
        }) >= 30.0:
            age_preference = "16_plus"
        elif self._max_score(scores_pct, {
            "misinformation_and_fake_news",
            "substance_abuse",
            "body_shaming",
            "sexist_content",
            "invasive_privacy_violation",
        }) >= 25.0:
            age_preference = "13_plus"

        warning_notes = [
            f"{category} ({score:.1f})"
            for category, score in sorted(scores_pct.items(), key=lambda x: x[1], reverse=True)
            if score >= 30.0
        ]

        return {
            "view_recommendation": view_recommendation,
            "age_preference": age_preference,
            "warning_notes": warning_notes[:6],
        }

    def _max_score(self, scores_pct: Dict[str, float], categories: set[str]) -> float:
        return max((scores_pct.get(category, 0.0) for category in categories), default=0.0)

    def _extract_transcript(self, path: Path) -> Optional[str]:
        try:
            whisper = importlib.import_module("whisper")
            model = whisper.load_model("base")
            result = model.transcribe(str(path))
            return result.get("text", "")
        except Exception:
            return None

    def _extract_ocr(self, path: Path) -> Optional[str]:
        try:
            pytesseract = importlib.import_module("pytesseract")
            from PIL import Image

            with Image.open(path) as image:
                return pytesseract.image_to_string(image)
        except Exception:
            return None
