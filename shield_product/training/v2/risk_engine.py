import math
from typing import Dict, List, Tuple

HARM_CUE_WORDS = (
    "hate",
    "violence",
    "attack",
    "extrem",
    "terror",
    "threat",
    "propaganda",
    "harass",
)

EVASION_PATTERNS = {
    "evade": 8.0,
    "dog whistle": 12.0,
    "coded": 8.0,
    "ironic": 4.0,
    "joking": 3.0,
    "not serious": 3.0,
}

MANIPULATION_PATTERNS = {
    "urgent": 6.0,
    "share now": 8.0,
    "wake up": 7.0,
    "before they delete": 9.0,
    "real truth": 8.0,
    "secret": 4.0,
}


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(value, upper))


def average_harm_score(category_scores: Dict[str, float]) -> float:
    if not category_scores:
        return 0.0

    keyed: List[float] = []
    for category, score in category_scores.items():
        category_lower = str(category).lower()
        if any(cue in category_lower for cue in HARM_CUE_WORDS):
            keyed.append(float(score))

    source = keyed if keyed else [float(score) for score in category_scores.values()]
    return round(sum(source) / max(len(source), 1), 4)


def compute_viral_harm_potential(
    safety_harm_score: float,
    engagement: Dict[str, float],
) -> float:
    likes = float(engagement.get("likes", 0.0))
    shares = float(engagement.get("shares", 0.0))
    comments = float(engagement.get("comments", 0.0))
    duets = float(engagement.get("duets", 0.0))
    stitches = float(engagement.get("stitches", 0.0))
    comment_sentiment = float(engagement.get("comment_sentiment", 0.0))

    reach_component = min((likes + 3.0 * shares + comments + 2.0 * (duets + stitches)) / 20000.0, 1.0)
    sentiment_amplifier = 1.0 + max(0.0, -comment_sentiment)
    potential = float(safety_harm_score) * (0.55 + 0.45 * reach_component) * sentiment_amplifier
    return round(_clamp(potential), 4)


def compute_cross_modal_contradiction(
    transcript_text: str,
    ocr_text: str,
    caption_text: str,
    frame_signals: Dict[str, float],
) -> float:
    text = " ".join([transcript_text, ocr_text, caption_text]).lower()

    positive_cues = sum(text.count(word) for word in ["happy", "love", "great", "safe", "calm", "fun"])
    negative_cues = sum(text.count(word) for word in ["kill", "hate", "attack", "fear", "panic", "danger"])
    text_valence = positive_cues - negative_cues

    motion = float(frame_signals.get("avg_motion", 0.0))
    brightness = float(frame_signals.get("avg_brightness", 0.0))
    visual_intensity = min((motion / 60.0) + max(0.0, (120.0 - brightness) / 120.0), 1.0)

    if text_valence > 0 and visual_intensity > 0.65:
        score = 40.0 + visual_intensity * 60.0
    elif text_valence < 0 and visual_intensity < 0.2:
        score = 35.0
    else:
        score = abs(text_valence) * 2.5 * visual_intensity

    return round(_clamp(score), 4)


def compute_adversarial_evasion_score(text: str) -> float:
    lowered = (text or "").lower()
    score = 0.0
    for cue, weight in EVASION_PATTERNS.items():
        score += lowered.count(cue) * weight
    return round(_clamp(score), 4)


def compute_cognitive_manipulation_index(
    metadata: Dict[str, object],
    frame_signals: Dict[str, float],
    engagement: Dict[str, float],
    text: str,
) -> float:
    lowered = (text or "").lower()
    pressure = 0.0
    for cue, weight in MANIPULATION_PATTERNS.items():
        pressure += lowered.count(cue) * weight

    cuts_per_minute = float(frame_signals.get("cuts_per_minute", 0.0))
    avg_motion = float(frame_signals.get("avg_motion", 0.0))
    watch_ratio = float(metadata.get("watch_ratio", 0.0) or 0.0)

    shares = float(engagement.get("shares", 0.0))
    comments = float(engagement.get("comments", 0.0))
    views = max(float(engagement.get("views", 0.0)), 1.0)

    rhythm_component = min((cuts_per_minute / 70.0) + (avg_motion / 120.0), 1.0)
    engagement_component = min(((shares + comments) / views) * 120.0, 1.0)
    pressure_component = min(pressure / 30.0, 1.0)
    retention_component = min(max(watch_ratio, 0.0), 1.0)

    score = 100.0 * (
        0.30 * rhythm_component
        + 0.30 * pressure_component
        + 0.20 * engagement_component
        + 0.20 * retention_component
    )
    return round(_clamp(score), 4)


def compute_network_diffusion_risk(
    engagement: Dict[str, float],
    safety_harm_score: float,
    contradiction_score: float,
    evasion_score: float,
) -> Dict[str, object]:
    likes = max(float(engagement.get("likes", 0.0)), 0.0)
    shares = max(float(engagement.get("shares", 0.0)), 0.0)
    comments = max(float(engagement.get("comments", 0.0)), 0.0)
    duets = max(float(engagement.get("duets", 0.0)), 0.0)
    stitches = max(float(engagement.get("stitches", 0.0)), 0.0)
    views = max(float(engagement.get("views", 0.0)), 1.0)
    comment_sentiment = float(engagement.get("comment_sentiment", 0.0))

    interaction_mass = likes + comments + 1.8 * shares + 2.0 * duets + 2.0 * stitches
    normalized_mass = min(math.log1p(interaction_mass) / math.log1p(1_000_000.0), 1.0)

    share_rate = min((shares / views) * 250.0, 1.0)
    remix_rate = min(((duets + stitches) / views) * 450.0, 1.0)
    cascade_pressure = min(0.55 * share_rate + 0.45 * remix_rate, 1.0)

    controversy = min(max(-comment_sentiment, 0.0), 1.0)
    content_activation = min(
        max(
            (0.55 * float(safety_harm_score) + 0.25 * float(contradiction_score) + 0.20 * float(evasion_score))
            / 100.0,
            0.0,
        ),
        1.0,
    )

    score = 100.0 * (
        0.35 * normalized_mass
        + 0.30 * cascade_pressure
        + 0.10 * controversy
        + 0.25 * content_activation
    )
    score = round(_clamp(score), 4)

    return {
        "score": score,
        "components": {
            "normalized_interaction_mass": round(normalized_mass, 4),
            "cascade_pressure": round(cascade_pressure, 4),
            "controversy": round(controversy, 4),
            "content_activation": round(content_activation, 4),
        },
        "formula": "100*(0.35*mass + 0.30*cascade + 0.10*controversy + 0.25*content_activation)",
    }


def compute_final_risk(
    harm_score: float,
    viral_harm: float,
    contradiction: float,
    evasion: float,
    cognitive: float,
    network_diffusion: float,
) -> float:
    final_risk = (
        0.35 * float(harm_score)
        + 0.20 * float(viral_harm)
        + 0.12 * float(contradiction)
        + 0.10 * float(evasion)
        + 0.10 * float(cognitive)
        + 0.13 * float(network_diffusion)
    )
    return round(_clamp(final_risk), 4)


def estimate_analysis_confidence(
    combined_text: str,
    metadata: Dict[str, object],
    engine_details: Dict[str, object],
) -> float:
    text_tokens = len((combined_text or "").split())
    text_score = min((text_tokens / 220.0) * 45.0, 45.0)

    diagnostics = metadata.get("extraction_diagnostics", {})
    available = diagnostics.get("available", {}) if isinstance(diagnostics, dict) else {}
    available_count = 0
    if isinstance(available, dict):
        available_count = sum(1 for value in available.values() if bool(value))
    extractor_score = min(available_count * 8.0, 24.0)

    engine_score = 0.0
    if bool(engine_details.get("used_embeddings")):
        engine_score += 12.0
    if bool(engine_details.get("used_chromadb")):
        engine_score += 10.0
    if bool(engine_details.get("used_groq")):
        engine_score += 9.0

    duration = _safe_float(metadata.get("duration"))
    duration_score = min(duration / 60.0 * 10.0, 10.0)

    confidence = text_score + extractor_score + engine_score + duration_score
    return round(_clamp(confidence), 4)


def infer_evidence_mode(confidence: float) -> Tuple[str, bool]:
    if confidence >= 70.0:
        return "full", False
    if confidence >= 35.0:
        return "fallback", False
    return "insufficient", True


def age_rating_from_risk(final_risk: float) -> str:
    if final_risk >= 80.0:
        return "18+"
    if final_risk >= 60.0:
        return "16+"
    if final_risk >= 40.0:
        return "13+"
    if final_risk >= 25.0:
        return "7+"
    return "U"


def verdict_from_risk(final_risk: float, network_diffusion: float, insufficient_evidence: bool) -> str:
    if insufficient_evidence:
        return "Insufficient evidence. Provide better media quality or enable more extractors."
    if final_risk >= 80.0 or network_diffusion >= 80.0:
        return "Severe risk detected. Block distribution and escalate for manual review."
    if final_risk >= 60.0 or network_diffusion >= 70.0:
        return "High risk detected. Restrict visibility and enforce strict guardrails."
    if final_risk >= 45.0:
        return "Moderate risk detected. Apply caution labels and age gating."
    return "Low risk signal. Content is acceptable with standard monitoring."


def top_categories(category_scores: Dict[str, float], top_k: int = 6) -> List[Dict[str, float]]:
    ranked = sorted(
        ({"category": str(category), "score": round(float(score), 4)} for category, score in category_scores.items()),
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    return ranked[: max(int(top_k), 1)]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
