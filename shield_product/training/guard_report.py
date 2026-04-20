from typing import Dict, List

from .guard_taxonomy import layer_to_categories


LAYER_MAP = layer_to_categories()


def build_guard_report(
    video_path: str,
    combined_scores: Dict[str, float],
    metadata: Dict[str, object],
    llm_verdict: str,
    phase1_harm_probability: float,
) -> Dict[str, object]:
    harm_categories = LAYER_MAP.get("Safety & Harm Layer", [])
    societal_categories = LAYER_MAP.get("Societal Impact Layer", [])
    positive_categories = LAYER_MAP.get("Positive Value Layer", [])

    harm_avg = _average(combined_scores, harm_categories)
    societal_avg = _average(combined_scores, societal_categories)
    positive_avg = _average(combined_scores, positive_categories)

    evidence = _evidence_quality(combined_scores, phase1_harm_probability, metadata)
    insufficient_evidence = evidence["insufficient"]
    confidence_value = float(evidence.get("confidence", 0.0))

    raw_overall_safety_score = max(
        0.0,
        min(
            100.0,
            100.0
            - (0.45 * harm_avg + 0.25 * societal_avg + 0.30 * phase1_harm_probability)
            + 0.20 * positive_avg,
        ),
    )
    raw_good_for_society = max(
        0.0,
        min(100.0, 0.7 * positive_avg + 0.3 * (100.0 - societal_avg)),
    )

    if insufficient_evidence:
        # When extraction/scoring produced no usable signal, avoid false certainty.
        overall_safety_score = 50.0
        good_for_society = 50.0
    else:
        # Blend low-confidence runs toward neutral to avoid overconfident extremes.
        confidence_weight = max(0.05, min(confidence_value, 1.0))
        overall_safety_score = 50.0 + (raw_overall_safety_score - 50.0) * confidence_weight
        good_for_society = 50.0 + (raw_good_for_society - 50.0) * confidence_weight

    overall_risk_score = max(0.0, min(100.0, 100.0 - overall_safety_score))

    strongest_risk = _max_score(combined_scores, harm_categories + societal_categories)
    age_rating = _age_rating(strongest_risk, combined_scores)
    safe_to_watch = (
        not insufficient_evidence
        and overall_safety_score >= 65.0
        and confidence_value >= 0.30
        and age_rating != "18+"
        and phase1_harm_probability < 60.0
    )

    warning_categories = [
        (name, value)
        for name, value in combined_scores.items()
        if value >= 45.0 and name not in positive_categories
    ]
    if phase1_harm_probability >= 50.0:
        warning_categories.append(("phase1_binary_harm_gate", phase1_harm_probability))
    if insufficient_evidence:
        warning_categories.append(("insufficient_evidence", 100.0))
    warning_categories.sort(key=lambda item: item[1], reverse=True)

    top_positive = sorted(
        ((name, combined_scores.get(name, 0.0)) for name in positive_categories),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    verdict = llm_verdict.strip() or (
        _insufficient_evidence_verdict(evidence)
        if insufficient_evidence
        else _default_verdict(
            overall_safety_score=overall_safety_score,
            warnings=warning_categories,
            positives=top_positive,
        )
    )

    return {
        "video_path": video_path,
        "overall_safety_score_out_of_100": round(overall_safety_score, 4),
        "overall_risk_score_out_of_100": round(overall_risk_score, 4),
        "content_age_rating": age_rating,
        "safe_to_watch": safe_to_watch,
        "good_for_society_percentage": round(good_for_society, 4),
        "analysis_confidence": round(evidence["confidence"] * 100.0, 4),
        "insufficient_evidence": insufficient_evidence,
        "evidence_mode": _evidence_mode(evidence),
        "phase1_harm_probability": round(phase1_harm_probability, 4),
        "layer_breakdown": {
            "Safety & Harm Layer": round(harm_avg, 4),
            "Societal Impact Layer": round(societal_avg, 4),
            "Positive Value Layer": round(positive_avg, 4),
        },
        "category_breakdown": combined_scores,
        "warning_notes": [f"{name}: {score:.1f}" for name, score in warning_categories[:8]],
        "top_positive_signals": [f"{name}: {score:.1f}" for name, score in top_positive],
        "metadata": metadata,
        "verdict": verdict,
    }


def _age_rating(strongest_risk: float, scores: Dict[str, float]) -> str:
    if strongest_risk >= 80.0:
        return "18+"
    if strongest_risk >= 60.0:
        return "13+"
    if strongest_risk >= 35.0:
        return "7+"
    return "U"


def _default_verdict(
    overall_safety_score: float,
    warnings: List[tuple[str, float]],
    positives: List[tuple[str, float]],
) -> str:
    if warnings:
        risk_text = ", ".join(f"{name} ({score:.1f})" for name, score in warnings[:3])
    else:
        risk_text = "low direct harm indicators"

    if positives:
        positive_text = ", ".join(f"{name} ({score:.1f})" for name, score in positives[:2])
    else:
        positive_text = "limited positive social value signals"

    return (
        f"Overall safety score is {overall_safety_score:.1f}/100. "
        f"Key risks: {risk_text}. "
        f"Positive signals: {positive_text}."
    )


def _insufficient_evidence_verdict(evidence: Dict[str, float | bool]) -> str:
    return (
        "Evidence quality is too low for a confident moderation decision. "
        f"Signal coverage: {float(evidence.get('signal_ratio', 0.0)) * 100.0:.1f}%. "
        "Enable transcription/OCR or provide caption/metadata sidecars for reliable scoring."
    )


def _evidence_mode(evidence: Dict[str, float | bool]) -> str:
    if bool(evidence.get("insufficient", False)):
        return "insufficient"
    if bool(evidence.get("fallback_only", False)):
        return "fallback"
    return "full"


def _evidence_quality(
    combined_scores: Dict[str, float],
    phase1_harm_probability: float,
    metadata: Dict[str, object],
) -> Dict[str, float | bool]:
    if not combined_scores:
        return {
            "signal_ratio": 0.0,
            "confidence": 0.0,
            "insufficient": True,
            "fallback_only": False,
            "has_primary_text_signal": False,
        }

    diagnostics = metadata.get("extraction_diagnostics", {}) if isinstance(metadata, dict) else {}
    source_presence = diagnostics.get("source_presence", {}) if isinstance(diagnostics, dict) else {}
    has_primary_text_signal = any(
        bool(source_presence.get(key, False))
        for key in ["has_transcript", "has_ocr_text", "has_caption_text", "has_hashtags"]
    )
    fallback_only = bool(diagnostics.get("fallback_context_applied", False)) and not has_primary_text_signal
    missing_components = diagnostics.get("missing_components", []) if isinstance(diagnostics, dict) else []
    missing_count = (
        len([item for item in missing_components if isinstance(item, str) and item.strip()])
        if isinstance(missing_components, list)
        else 0
    )

    total = len(combined_scores)
    nonzero = sum(1 for value in combined_scores.values() if float(value) >= 1.0)
    strong = sum(1 for value in combined_scores.values() if float(value) >= 8.0)
    signal_ratio = nonzero / max(total, 1)

    phase_signal = max(0.0, min(float(phase1_harm_probability), 100.0)) / 100.0

    text_lengths = diagnostics.get("text_lengths", {}) if isinstance(diagnostics, dict) else {}
    combined_chars = _safe_float(text_lengths.get("combined_chars", 0.0))
    text_depth = min(combined_chars / 1500.0, 1.0)

    primary_sources_total = 4
    primary_sources_present = sum(
        1
        for key in ["has_transcript", "has_ocr_text", "has_caption_text", "has_hashtags"]
        if bool(source_presence.get(key, False))
    )
    source_coverage = primary_sources_present / max(primary_sources_total, 1)

    available = diagnostics.get("available", {}) if isinstance(diagnostics, dict) else {}
    if isinstance(available, dict):
        extractor_ready_count = sum(1 for value in available.values() if bool(value))
        extractor_health = extractor_ready_count / max(len(available), 1)
    else:
        extractor_health = 0.0

    confidence = min(
        1.0,
        0.45 * text_depth + 0.25 * source_coverage + 0.20 * extractor_health + 0.10 * phase_signal,
    )

    if missing_count > 0:
        confidence *= max(0.4, 1.0 - 0.12 * missing_count)

    if fallback_only:
        # Fallback-only text can be noisy. Cap confidence unless phase-1 adds strong evidence.
        confidence = min(confidence * 0.35, 0.20 + 0.45 * phase_signal)

    insufficient = strong == 0 and phase1_harm_probability <= 0.0 and signal_ratio < 0.08
    if fallback_only:
        # Without transcript/OCR/caption/hashtags, require independent phase-1 confidence.
        insufficient = phase_signal < 0.40
    elif not has_primary_text_signal and strong < 2 and phase_signal < 0.25:
        insufficient = True

    return {
        "signal_ratio": signal_ratio,
        "confidence": confidence,
        "insufficient": insufficient,
        "fallback_only": fallback_only,
        "has_primary_text_signal": has_primary_text_signal,
    }


def _average(scores: Dict[str, float], categories: List[str]) -> float:
    if not categories:
        return 0.0
    values = [scores.get(category, 0.0) for category in categories]
    return sum(values) / len(values)


def _max_score(scores: Dict[str, float], categories: List[str]) -> float:
    if not categories:
        return 0.0
    return max(scores.get(category, 0.0) for category in categories)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# Backward compatibility for legacy imports.
build_reelguard_report = build_guard_report
