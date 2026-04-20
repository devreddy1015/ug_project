import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .guard_taxonomy import layer_to_categories


LAYER_MAP = layer_to_categories()


def load_engagement_metadata(video_path: Path) -> Dict[str, float | str]:
    candidates = [
        video_path.with_suffix(".engagement.json"),
        video_path.with_suffix(".meta.json"),
        video_path.with_suffix(".json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if not isinstance(payload, dict):
                continue
            return {
                "likes": float(payload.get("likes", 0.0)),
                "shares": float(payload.get("shares", 0.0)),
                "comments": float(payload.get("comments", 0.0)),
                "duets": float(payload.get("duets", payload.get("duet_chain", 0.0))),
                "stitches": float(payload.get("stitches", payload.get("stitch_chain", 0.0))),
                "views": float(payload.get("views", 0.0)),
                "comment_sentiment": float(payload.get("comment_sentiment", 0.0)),
                "creator_id": str(payload.get("creator_id", "unknown_creator")),
                "region": str(payload.get("region", "global")),
            }
        except Exception:
            continue
    return {
        "likes": 0.0,
        "shares": 0.0,
        "comments": 0.0,
        "duets": 0.0,
        "stitches": 0.0,
        "views": 0.0,
        "comment_sentiment": 0.0,
        "creator_id": "unknown_creator",
        "region": "global",
    }


def aggregate_temporal_scores(
    base_scores: Dict[str, float],
    temporal_scores: List[Dict[str, object]],
) -> Dict[str, float]:
    if not temporal_scores:
        return dict(base_scores)

    categories = list(base_scores.keys())
    weighted_sum = {category: 0.0 for category in categories}
    total_weight = 0.0

    for item in temporal_scores:
        scores = item.get("scores", {})
        weight = float(item.get("recency_weight", 1.0))
        if weight <= 0:
            continue
        total_weight += weight
        for category in categories:
            weighted_sum[category] += weight * float(scores.get(category, 0.0))

    if total_weight <= 0:
        return dict(base_scores)

    merged: Dict[str, float] = {}
    for category in categories:
        temporal_value = weighted_sum[category] / total_weight
        merged[category] = round(
            0.65 * float(base_scores.get(category, 0.0)) + 0.35 * temporal_value, 4
        )
    return merged


def timestamp_attribution(
    temporal_scores: List[Dict[str, object]], top_k: int = 6
) -> List[Dict[str, object]]:
    if not temporal_scores:
        return []

    peaks: List[Dict[str, object]] = []
    categories = set()
    for entry in temporal_scores:
        categories.update(entry.get("scores", {}).keys())

    for category in categories:
        best = max(
            temporal_scores,
            key=lambda item: float(item.get("scores", {}).get(category, 0.0)),
        )
        score = float(best.get("scores", {}).get(category, 0.0))
        if score < 30.0:
            continue
        peaks.append(
            {
                "category": category,
                "score": round(score, 4),
                "start": best.get("start"),
                "end": best.get("end"),
            }
        )

    peaks.sort(key=lambda item: float(item["score"]), reverse=True)
    return peaks[:top_k]


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
    potential = safety_harm_score * (0.55 + 0.45 * reach_component) * sentiment_amplifier
    return round(max(0.0, min(potential, 100.0)), 4)


def compute_network_diffusion_risk(
    engagement: Dict[str, float],
    safety_harm_score: float,
    contradiction_score: float,
    evasion_score: float,
) -> Dict[str, object]:
    """Approximate diffusion risk using engagement as a social-graph proxy."""
    likes = max(float(engagement.get("likes", 0.0)), 0.0)
    shares = max(float(engagement.get("shares", 0.0)), 0.0)
    comments = max(float(engagement.get("comments", 0.0)), 0.0)
    duets = max(float(engagement.get("duets", 0.0)), 0.0)
    stitches = max(float(engagement.get("stitches", 0.0)), 0.0)
    views = max(float(engagement.get("views", 0.0)), 1.0)
    comment_sentiment = float(engagement.get("comment_sentiment", 0.0))

    # Network pressure proxies: local cascade tendency and normalized interaction mass.
    interaction_mass = (likes + comments + 1.8 * shares + 2.0 * duets + 2.0 * stitches)
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
    score = round(max(0.0, min(score, 100.0)), 4)

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

    return round(max(0.0, min(score, 100.0)), 4)


def apply_cultural_adapter(scores: Dict[str, float], region: str) -> Dict[str, float]:
    region_key = (region or "global").lower()
    adapters = {
        "global": {},
        "south_asia": {
            "political_propaganda": 1.1,
            "religious_extremism": 1.15,
        },
        "mena": {
            "extremist_content": 1.1,
            "political_propaganda": 1.08,
        },
        "eu": {
            "privacy_violation": 1.12,
            "misinformation": 1.08,
        },
        "us": {
            "financial_scams": 1.1,
            "misinformation": 1.1,
        },
    }
    weights = adapters.get(region_key, adapters["global"])

    adjusted = dict(scores)
    for category, factor in weights.items():
        if category in adjusted:
            adjusted[category] = round(max(0.0, min(adjusted[category] * factor, 100.0)), 4)
    return adjusted


def adversarial_evasion_score(text: str) -> float:
    lowered = (text or "").lower()

    # Focus on in-word obfuscation patterns and normalize by token mass so normal prose
    # does not saturate the score.
    tokens = re.findall(r"[a-z0-9@#$%^&*_.+\-]{3,}", lowered)
    token_count = max(len(tokens), 1)

    masked_tokens = sum(
        1
        for token in tokens
        if re.search(r"[a-z](?:[._*\-][a-z]){2,}", token)
    )
    leetspeak_tokens = sum(
        1
        for token in tokens
        if re.search(r"[a-z][0-9]|[0-9][a-z]", token) and re.search(r"[01345789]", token)
    )
    repeated_symbol_clusters = len(re.findall(r"[@#$%^&*_+=]{3,}", lowered))
    punctuation_spam = len(re.findall(r"([!?.,])\1{2,}", lowered))

    raw_signal = (
        22.0 * masked_tokens
        + 8.0 * leetspeak_tokens
        + 12.0 * repeated_symbol_clusters
        + 5.0 * punctuation_spam
    )
    normalized = raw_signal / max(token_count * 1.8, 1.0)
    score = 100.0 * (1.0 - math.exp(-normalized))
    return round(max(0.0, min(score, 100.0)), 4)


def cognitive_manipulation_index(
    metadata: Dict[str, object], frame_signals: Dict[str, float], engagement: Dict[str, float]
) -> float:
    duration = float(metadata.get("duration") or 0.0)
    sampled_frames = float(frame_signals.get("sampled_frames", 0.0))
    motion = float(frame_signals.get("avg_motion", 0.0))
    brightness = float(frame_signals.get("avg_brightness", 0.0))

    cuts_per_second = sampled_frames / max(duration, 1.0)
    pace = min(cuts_per_second / 0.8, 1.0)
    motion_intensity = min(motion / 70.0, 1.0)
    sensory_stress = min(max(0.0, (140.0 - brightness) / 140.0), 1.0)

    interaction_pressure = min(
        (float(engagement.get("shares", 0.0)) + float(engagement.get("duets", 0.0))) / 5000.0,
        1.0,
    )

    index = 100.0 * (0.35 * pace + 0.30 * motion_intensity + 0.20 * sensory_stress + 0.15 * interaction_pressure)
    return round(max(0.0, min(index, 100.0)), 4)


def societal_benefit_index(category_breakdown: Dict[str, float]) -> float:
    positive_categories = LAYER_MAP.get("Positive Value Layer", [])
    societal_categories = LAYER_MAP.get("Societal Impact Layer", [])

    positive = _average(category_breakdown, positive_categories)
    societal_harm = _average(category_breakdown, societal_categories)
    factual_accuracy_proxy = 100.0 - float(category_breakdown.get("misinformation", 0.0))

    sbi = 0.45 * positive + 0.30 * factual_accuracy_proxy + 0.25 * (100.0 - societal_harm)
    return round(max(0.0, min(sbi, 100.0)), 4)


def societal_benefit_details(
    category_breakdown: Dict[str, float], cognitive_manipulation_index_value: float
) -> Dict[str, object]:
    positive_categories = LAYER_MAP.get("Positive Value Layer", [])
    societal_categories = LAYER_MAP.get("Societal Impact Layer", [])

    educational_density = float(category_breakdown.get("educational_value", 0.0))
    emotional_valence = float(category_breakdown.get("emotional_positivity", 0.0))
    community_cohesion = float(category_breakdown.get("community_building", 0.0))
    creativity = float(category_breakdown.get("creativity", 0.0))
    positive_mean = _average(category_breakdown, positive_categories)
    societal_harm = _average(category_breakdown, societal_categories)
    factual_accuracy = 100.0 - float(category_breakdown.get("misinformation", 0.0))

    sbi_score = (
        0.20 * educational_density
        + 0.15 * emotional_valence
        + 0.15 * community_cohesion
        + 0.10 * creativity
        + 0.20 * positive_mean
        + 0.20 * factual_accuracy
        - 0.15 * societal_harm
        - 0.10 * float(cognitive_manipulation_index_value)
    )
    sbi_score = round(max(0.0, min(sbi_score, 100.0)), 4)

    return {
        "sbi_score": sbi_score,
        "components": {
            "educational_density": round(educational_density, 4),
            "emotional_valence": round(emotional_valence, 4),
            "community_cohesion": round(community_cohesion, 4),
            "creativity": round(creativity, 4),
            "positive_mean": round(positive_mean, 4),
            "factual_accuracy_proxy": round(factual_accuracy, 4),
            "societal_harm": round(societal_harm, 4),
            "cognitive_manipulation_index": round(float(cognitive_manipulation_index_value), 4),
        },
        "formula": "0.20*educational + 0.15*emotional + 0.15*community + 0.10*creativity + 0.20*positive_mean + 0.20*factual_accuracy - 0.15*societal_harm - 0.10*cognitive_index",
    }


def update_creator_profile(
    profile_path: Path,
    creator_id: str,
    video_path: str,
    safety_score: float,
    harm_probability: float,
) -> Dict[str, object]:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    state = _load_json(profile_path)

    creator_entry = state.setdefault(
        creator_id,
        {
            "videos": [],
            "avg_safety_score": 0.0,
            "avg_harm_probability": 0.0,
            "risk_trend": "stable",
            "total_videos": 0,
        },
    )

    creator_entry["videos"].append(
        {
            "video_path": video_path,
            "safety_score": round(safety_score, 4),
            "harm_probability": round(harm_probability, 4),
        }
    )
    creator_entry["videos"] = creator_entry["videos"][-50:]

    total = len(creator_entry["videos"])
    safety_values = [float(item["safety_score"]) for item in creator_entry["videos"]]
    harm_values = [float(item["harm_probability"]) for item in creator_entry["videos"]]

    creator_entry["avg_safety_score"] = round(sum(safety_values) / max(total, 1), 4)
    creator_entry["avg_harm_probability"] = round(sum(harm_values) / max(total, 1), 4)
    creator_entry["total_videos"] = total
    creator_entry["risk_trend"] = _risk_trend(harm_values)

    _write_json(profile_path, state)
    return creator_entry


def export_federated_update(
    output_path: Path,
    creator_id: str,
    video_path: str,
    category_breakdown: Dict[str, float],
    phase1_harm_probability: float,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "creator_id": creator_id,
        "video_path_hash": str(abs(hash(video_path))),
        "phase1_harm_probability": round(float(phase1_harm_probability), 4),
        "category_breakdown": {
            key: round(float(value), 4) for key, value in category_breakdown.items()
        },
    }

    existing: List[Dict[str, object]] = []
    if output_path.exists():
        try:
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            existing = []
    existing.append(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _risk_trend(values: List[float]) -> str:
    if len(values) < 5:
        return "insufficient_history"
    first = sum(values[: len(values) // 2]) / max(len(values) // 2, 1)
    second = sum(values[len(values) // 2 :]) / max(len(values) - len(values) // 2, 1)
    delta = second - first
    if delta > 8.0:
        return "escalating"
    if delta < -8.0:
        return "improving"
    return "stable"


def _average(scores: Dict[str, float], categories: List[str]) -> float:
    if not categories:
        return 0.0
    values = [float(scores.get(category, 0.0)) for category in categories]
    return sum(values) / len(values)


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return {}
    return {}


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
