from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from .adapters import (
    LegacyCulturalAdapter,
    LegacyEngagementProvider,
    LegacySemanticScorer,
    LegacySignalExtractor,
)
from .config import GuardV2Config
from .risk_engine import (
    age_rating_from_risk,
    average_harm_score,
    compute_adversarial_evasion_score,
    compute_cognitive_manipulation_index,
    compute_cross_modal_contradiction,
    compute_final_risk,
    compute_network_diffusion_risk,
    compute_viral_harm_potential,
    estimate_analysis_confidence,
    infer_evidence_mode,
    top_categories,
    verdict_from_risk,
)
from .types import MediaAsset, SemanticResult, SignalPayload


class SignalExtractor(Protocol):
    def extract(
        self,
        video_path: Path,
        *,
        frame_count: int,
        enable_whisper: bool,
        enable_ocr: bool,
    ) -> SignalPayload:
        ...


class SemanticScorer(Protocol):
    def score(self, combined_text: str) -> SemanticResult:
        ...


class EngagementProvider(Protocol):
    def load(self, video_path: Path) -> Dict[str, object]:
        ...


class CulturalAdapter(Protocol):
    def adapt(self, category_scores: Dict[str, float], region: str) -> Dict[str, float]:
        ...


class GuardV2Pipeline:
    def __init__(
        self,
        config: GuardV2Config,
        signal_extractor: Optional[SignalExtractor] = None,
        semantic_scorer: Optional[SemanticScorer] = None,
        engagement_provider: Optional[EngagementProvider] = None,
        cultural_adapter: Optional[CulturalAdapter] = None,
    ) -> None:
        self.config = config
        self.signal_extractor = signal_extractor or LegacySignalExtractor()
        self.semantic_scorer = semantic_scorer or LegacySemanticScorer()
        self.engagement_provider = engagement_provider or LegacyEngagementProvider()
        self.cultural_adapter = cultural_adapter or LegacyCulturalAdapter()

    def analyze_assets(
        self,
        assets: List[MediaAsset],
        *,
        fail_on_item_error: bool = False,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        completed: List[Dict[str, object]] = []
        failed: List[Dict[str, object]] = []

        for asset in assets:
            try:
                completed.append(self.analyze_asset(asset))
            except Exception as error:
                failed.append(self._build_failed_result(asset, error))
                if fail_on_item_error:
                    raise

        return completed, failed

    def analyze_asset(self, asset: MediaAsset) -> Dict[str, object]:
        signals = self.signal_extractor.extract(
            asset.path,
            frame_count=int(self.config.frame_count),
            enable_whisper=bool(self.config.enable_whisper),
            enable_ocr=bool(self.config.enable_ocr),
        )

        semantic = self.semantic_scorer.score(signals.combined_text)
        engagement = self.engagement_provider.load(asset.path)

        region = self._resolve_region(self.config.region, engagement)
        category_scores = self.cultural_adapter.adapt(semantic.combined_scores, region)

        harm_score = average_harm_score(category_scores)
        viral_harm = compute_viral_harm_potential(harm_score, engagement)
        contradiction = compute_cross_modal_contradiction(
            signals.transcript_text,
            signals.ocr_text,
            signals.caption_text,
            signals.frame_signals,
        )
        evasion = compute_adversarial_evasion_score(signals.combined_text)
        cognitive = compute_cognitive_manipulation_index(
            metadata=signals.metadata,
            frame_signals=signals.frame_signals,
            engagement=engagement,
            text=signals.combined_text,
        )
        network_diffusion = compute_network_diffusion_risk(
            engagement=engagement,
            safety_harm_score=harm_score,
            contradiction_score=contradiction,
            evasion_score=evasion,
        )

        final_risk = compute_final_risk(
            harm_score=harm_score,
            viral_harm=viral_harm,
            contradiction=contradiction,
            evasion=evasion,
            cognitive=cognitive,
            network_diffusion=float(network_diffusion["score"]),
        )

        confidence = estimate_analysis_confidence(
            combined_text=signals.combined_text,
            metadata=signals.metadata,
            engine_details=semantic.engine_details,
        )
        evidence_mode, insufficient_evidence = infer_evidence_mode(confidence)

        if insufficient_evidence:
            # Pull risky outliers toward neutral when evidence is too weak.
            final_risk = round((0.45 * final_risk) + (0.55 * 50.0), 4)

        safety_score = round(max(0.0, 100.0 - final_risk), 4)
        society_score = round(max(0.0, 100.0 - (0.70 * final_risk + 0.30 * cognitive)), 4)
        age_rating = age_rating_from_risk(final_risk)

        safe_to_watch = bool(
            final_risk < float(self.config.final_block_threshold)
            and float(network_diffusion["score"]) < float(self.config.network_block_threshold)
            and not insufficient_evidence
        )

        verdict = verdict_from_risk(
            final_risk=final_risk,
            network_diffusion=float(network_diffusion["score"]),
            insufficient_evidence=insufficient_evidence,
        )

        metadata = {
            **signals.metadata,
            "frame_signals": signals.frame_signals,
            "hashtags": signals.hashtags,
            "caption_text": signals.caption_text,
            "engagement": engagement,
            "region": region,
        }

        return {
            "processing_status": "completed",
            "error_message": None,
            "video_path": str(asset.path),
            "source_id": asset.source_id,
            "region": region,
            "overall_risk_score_out_of_100": final_risk,
            "overall_safety_score_out_of_100": safety_score,
            "good_for_society_percentage": society_score,
            "safe_to_watch": safe_to_watch,
            "content_age_rating": age_rating,
            "phase1_harm_probability": 0.0,
            "viral_harm_potential": round(viral_harm, 4),
            "cross_modal_contradiction_score": round(contradiction, 4),
            "adversarial_evasion_score": round(evasion, 4),
            "cognitive_manipulation_index": round(cognitive, 4),
            "network_diffusion_risk": round(float(network_diffusion["score"]), 4),
            "network_diffusion_details": network_diffusion["components"],
            "analysis_confidence": round(confidence, 4),
            "insufficient_evidence": insufficient_evidence,
            "evidence_mode": evidence_mode,
            "category_breakdown": {key: round(float(value), 4) for key, value in category_scores.items()},
            "top_risk_categories": top_categories(category_scores, top_k=int(self.config.top_categories)),
            "llm_scores": semantic.llm_scores,
            "vector_scores": semantic.vector_scores,
            "engine_details": semantic.engine_details,
            "verdict": verdict,
            "llm_verdict": semantic.llm_verdict,
            "metadata": metadata,
        }

    def _resolve_region(self, region_arg: str, engagement: Dict[str, object]) -> str:
        if str(region_arg).lower() != "auto":
            return str(region_arg)

        for key in ("region", "geo_region", "locale_region"):
            value = engagement.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return "global"

    def _build_failed_result(self, asset: MediaAsset, error: Exception) -> Dict[str, object]:
        message = str(error).strip() or "Unknown processing failure"
        if len(message) > 500:
            message = message[:500] + "..."

        return {
            "processing_status": "failed",
            "error_message": message,
            "video_path": str(asset.path),
            "source_id": asset.source_id,
            "region": "unknown",
            "overall_risk_score_out_of_100": 50.0,
            "overall_safety_score_out_of_100": 50.0,
            "good_for_society_percentage": 50.0,
            "safe_to_watch": False,
            "content_age_rating": "U",
            "phase1_harm_probability": 0.0,
            "viral_harm_potential": 0.0,
            "cross_modal_contradiction_score": 0.0,
            "adversarial_evasion_score": 0.0,
            "cognitive_manipulation_index": 0.0,
            "network_diffusion_risk": 0.0,
            "network_diffusion_details": {},
            "analysis_confidence": 0.0,
            "insufficient_evidence": True,
            "evidence_mode": "failed",
            "category_breakdown": {},
            "top_risk_categories": [],
            "llm_scores": {},
            "vector_scores": {},
            "engine_details": {},
            "verdict": f"Analysis failed: {message}",
            "llm_verdict": "",
            "metadata": {},
        }
