from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Optional

from training.v2.adapters import LegacySemanticScorer
from training.v2.config import GuardV2Config
from training.v2.pipeline import GuardV2Pipeline
from training.v2.risk_engine import (
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
from training.v2.types import MediaAsset

from .config import SETTINGS


CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_COOLDOWN_SECONDS = 120


_DEFAULT_CATEGORIES = {
    "explicit_content",
    "violence",
    "self_harm",
    "hate_speech",
    "substance_abuse",
    "child_safety_risk",
    "cyberbullying",
    "harassment",
    "sexual_predation",
    "extremist_content",
    "misinformation",
    "body_image_harm",
    "financial_scams",
    "political_propaganda",
    "mental_health_triggers",
    "addiction_bait",
    "privacy_violation",
    "gambling_promotion",
    "manipulative_marketing",
    "polarization",
    "educational_value",
    "emotional_positivity",
    "cultural_appreciation",
    "creativity",
    "community_building",
    "empathy_support",
    "constructive_dialogue",
    "skill_development",
}

_CATEGORY_KEYWORDS = {
    "explicit_content": ("nsfw", "explicit", "porn", "sexual"),
    "violence": ("kill", "attack", "blood", "shoot", "murder"),
    "self_harm": ("suicide", "self harm", "cut myself", "overdose"),
    "hate_speech": ("hate", "racial slur", "inferior race"),
    "substance_abuse": ("drug", "meth", "cocaine", "overdose", "high"),
    "child_safety_risk": ("minor", "underage", "child abuse"),
    "cyberbullying": ("bully", "harass online", "dogpile"),
    "harassment": ("harass", "threaten", "intimidate"),
    "sexual_predation": ("groom", "predator", "coerce"),
    "extremist_content": ("extremist", "terror", "radicalize"),
    "misinformation": ("fake news", "hoax", "conspiracy"),
    "body_image_harm": ("fat shame", "body shame", "ugly"),
    "financial_scams": ("crypto scam", "ponzi", "wire money", "fraud"),
    "political_propaganda": ("propaganda", "regime", "state media"),
    "mental_health_triggers": ("panic", "depression", "trauma trigger"),
    "addiction_bait": ("lootbox", "endless scroll", "dopamine"),
    "privacy_violation": ("dox", "leak address", "private photos"),
    "gambling_promotion": ("bet now", "casino", "sportsbook"),
    "manipulative_marketing": ("buy now", "limited time", "fear of missing out"),
    "polarization": ("us vs them", "civil war", "enemy camp"),
    "educational_value": ("learn", "tutorial", "explain", "lesson"),
    "emotional_positivity": ("support", "encourage", "healing", "kindness"),
    "cultural_appreciation": ("heritage", "culture", "tradition"),
    "creativity": ("art", "creative", "music", "design"),
    "community_building": ("community", "together", "volunteer"),
    "empathy_support": ("empathy", "listen", "support group"),
    "constructive_dialogue": ("dialogue", "debate respectfully", "common ground"),
    "skill_development": ("practice", "skill", "improve", "training"),
}


_VIDEO_PIPELINE: Optional[GuardV2Pipeline] = None
_SEMANTIC_SCORER: Optional[LegacySemanticScorer] = None
_CIRCUIT_FAILURES = 0
_CIRCUIT_OPEN_UNTIL = 0.0


def _build_video_pipeline() -> GuardV2Pipeline:
    config = GuardV2Config(
        frame_count=int(SETTINGS.guard_frame_count),
        enable_whisper=bool(SETTINGS.guard_enable_whisper),
        enable_ocr=bool(SETTINGS.guard_enable_ocr),
        region=str(SETTINGS.guard_region),
    )
    return GuardV2Pipeline(config)


def _get_video_pipeline() -> GuardV2Pipeline:
    global _VIDEO_PIPELINE
    if _VIDEO_PIPELINE is None:
        _VIDEO_PIPELINE = _build_video_pipeline()
    return _VIDEO_PIPELINE


def _get_semantic_scorer() -> LegacySemanticScorer:
    global _SEMANTIC_SCORER
    if _SEMANTIC_SCORER is None:
        _SEMANTIC_SCORER = LegacySemanticScorer()
    return _SEMANTIC_SCORER


def _circuit_allows_semantic() -> bool:
    global _CIRCUIT_OPEN_UNTIL
    if _CIRCUIT_OPEN_UNTIL <= 0.0:
        return True
    if time.monotonic() >= _CIRCUIT_OPEN_UNTIL:
        _CIRCUIT_OPEN_UNTIL = 0.0
        return True
    return False


def _record_semantic_success() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    _CIRCUIT_FAILURES = 0
    _CIRCUIT_OPEN_UNTIL = 0.0


def _record_semantic_failure() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    _CIRCUIT_FAILURES += 1
    if _CIRCUIT_FAILURES >= CIRCUIT_FAILURE_THRESHOLD:
        _CIRCUIT_OPEN_UNTIL = time.monotonic() + float(CIRCUIT_COOLDOWN_SECONDS)


def _score_with_keyword_fallback(text: str) -> Dict[str, float]:
    lowered = (text or "").lower()
    scores: Dict[str, float] = {}

    for category in _DEFAULT_CATEGORIES:
        scores[category] = 0.0

    for category, keywords in _CATEGORY_KEYWORDS.items():
        hits = 0
        for keyword in keywords:
            hits += lowered.count(keyword)
        scores[category] = min(float(hits) * 24.0, 100.0)

    return scores


def analyze_video_file(path: Path) -> Dict[str, Any]:
    asset = MediaAsset(path=path, source_id=f"video_{path.stem}")
    report = _get_video_pipeline().analyze_asset(asset)
    report["modality"] = "video"
    report["analysis_mode"] = "guard_v2_pipeline"
    return report


def analyze_text_payload(text: str) -> Dict[str, Any]:
    limited = str(text)[: int(SETTINGS.text_max_chars)]
    metadata = {
        "duration": 0.0,
        "extraction_diagnostics": {
            "available": {
                "direct_text_input": True,
                "semantic_engine": True,
            }
        },
    }
    return _analyze_semantic_payload(
        combined_text=limited,
        source_id="text_inline",
        modality="text",
        metadata=metadata,
    )


def analyze_non_video_file(path: Path, modality: str) -> Dict[str, Any]:
    context_chunks = [path.name]

    sidecar = path.with_suffix(".txt")
    if sidecar.exists() and sidecar.is_file():
        try:
            sidecar_text = sidecar.read_text(encoding="utf-8", errors="ignore")
            if sidecar_text.strip():
                context_chunks.append(sidecar_text[: int(SETTINGS.text_max_chars)])
        except OSError:
            pass

    combined_text = "\n".join(chunk for chunk in context_chunks if chunk).strip()
    metadata = {
        "duration": 0.0,
        "file_size_bytes": path.stat().st_size,
        "file_name": path.name,
        "extraction_diagnostics": {
            "available": {
                "file_context": True,
                "semantic_engine": True,
                "sidecar_text": sidecar.exists(),
            }
        },
    }

    return _analyze_semantic_payload(
        combined_text=combined_text,
        source_id=f"{modality}_{path.stem}",
        modality=modality,
        metadata=metadata,
        video_path=str(path),
    )


def _analyze_semantic_payload(
    *,
    combined_text: str,
    source_id: str,
    modality: str,
    metadata: Dict[str, Any],
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    category_scores: Dict[str, float]
    llm_scores: Dict[str, float]
    vector_scores: Dict[str, float]
    llm_verdict = ""
    engine_details: Dict[str, Any]

    if _circuit_allows_semantic():
        try:
            semantic = _get_semantic_scorer().score(combined_text)
            category_scores = {str(k): float(v) for k, v in semantic.combined_scores.items()}
            llm_scores = {str(k): float(v) for k, v in semantic.llm_scores.items()}
            vector_scores = {str(k): float(v) for k, v in semantic.vector_scores.items()}
            llm_verdict = str(semantic.llm_verdict)
            engine_details = dict(semantic.engine_details)
            _record_semantic_success()
        except Exception:
            _record_semantic_failure()
            category_scores = _score_with_keyword_fallback(combined_text)
            llm_scores = {key: 0.0 for key in category_scores.keys()}
            vector_scores = dict(category_scores)
            llm_verdict = "semantic engine unavailable; keyword fallback applied"
            engine_details = {
                "used_groq": False,
                "used_embeddings": False,
                "used_chromadb": False,
                "circuit_open": True,
            }
    else:
        category_scores = _score_with_keyword_fallback(combined_text)
        llm_scores = {key: 0.0 for key in category_scores.keys()}
        vector_scores = dict(category_scores)
        llm_verdict = "semantic engine circuit open; keyword fallback applied"
        engine_details = {
            "used_groq": False,
            "used_embeddings": False,
            "used_chromadb": False,
            "circuit_open": True,
        }

    for category in _DEFAULT_CATEGORIES:
        category_scores.setdefault(category, 0.0)
        llm_scores.setdefault(category, 0.0)
        vector_scores.setdefault(category, 0.0)

    engagement: Dict[str, float] = {}
    region = "global"

    harm_score = average_harm_score(category_scores)
    viral_harm = compute_viral_harm_potential(harm_score, engagement)
    contradiction = compute_cross_modal_contradiction("", "", combined_text, {})
    evasion = compute_adversarial_evasion_score(combined_text)
    cognitive = compute_cognitive_manipulation_index(
        metadata=metadata,
        frame_signals={},
        engagement=engagement,
        text=combined_text,
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
        combined_text=combined_text,
        metadata=metadata,
        engine_details=engine_details,
    )
    evidence_mode, insufficient_evidence = infer_evidence_mode(confidence)

    if insufficient_evidence:
        final_risk = round((0.45 * final_risk) + (0.55 * 50.0), 4)

    safety_score = round(max(0.0, 100.0 - final_risk), 4)
    society_score = round(max(0.0, 100.0 - (0.70 * final_risk + 0.30 * cognitive)), 4)
    age_rating = age_rating_from_risk(final_risk)

    safe_to_watch = bool(
        final_risk < 70.0
        and float(network_diffusion["score"]) < 70.0
        and not insufficient_evidence
    )

    verdict = verdict_from_risk(
        final_risk=final_risk,
        network_diffusion=float(network_diffusion["score"]),
        insufficient_evidence=insufficient_evidence,
    )

    return {
        "processing_status": "completed",
        "error_message": None,
        "video_path": video_path,
        "source_id": source_id,
        "modality": modality,
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
        "top_risk_categories": top_categories(category_scores, top_k=6),
        "llm_scores": llm_scores,
        "vector_scores": vector_scores,
        "engine_details": engine_details,
        "verdict": verdict,
        "llm_verdict": llm_verdict,
        "metadata": {
            **metadata,
            "region": region,
            "engagement": engagement,
        },
        "analysis_mode": "guard_semantic_service",
    }
