from dataclasses import dataclass
from typing import Dict, List, Tuple

from Engine.utils import probe_media


@dataclass(frozen=True)
class SegmentScore:
    start_sec: float
    end_sec: float
    max_risk_score: float
    dominant_categories: List[str]
    category_scores: Dict[str, float]


def temporal_segment_analysis(
    video_path,
    transcript_text: str,
    caption_text: str,
    hashtags: List[str],
    vector_scorer,
    window_sec: float = 5.0,
    recency_bias: float = 1.25,
    max_segments: int = 24,
) -> Tuple[List[SegmentScore], Dict[str, float], List[Dict[str, object]]]:
    duration = _safe_float(probe_media(video_path).get("duration"))
    boundaries = _build_boundaries(duration, window_sec=window_sec, max_segments=max_segments)

    tokens = transcript_text.split()
    segment_count = max(len(boundaries), 1)
    tags_text = " ".join(f"#{tag}" for tag in hashtags)
    score_cache: Dict[str, Dict[str, float]] = {}

    segments: List[SegmentScore] = []
    weighted_sum: Dict[str, float] = {}
    weight_total = 0.0

    for idx, (start_sec, end_sec) in enumerate(boundaries):
        transcript_chunk = _slice_tokens(tokens, idx, segment_count)
        segment_text = "\n".join(part for part in [transcript_chunk, caption_text, tags_text] if part)
        category_scores = score_cache.get(segment_text)
        if category_scores is None:
            category_scores = vector_scorer(segment_text)
            score_cache[segment_text] = category_scores

        dominant_categories = [
            name
            for name, score in sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
            if score >= 35.0
        ][:3]
        max_risk = max(category_scores.values()) if category_scores else 0.0

        segments.append(
            SegmentScore(
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                max_risk_score=round(max_risk, 4),
                dominant_categories=dominant_categories,
                category_scores=category_scores,
            )
        )

        # Later segments carry stronger weight to capture escalation patterns.
        position = (idx + 1) / max(len(boundaries), 1)
        weight = 1.0 + recency_bias * position
        weight_total += weight
        for category, value in category_scores.items():
            weighted_sum[category] = weighted_sum.get(category, 0.0) + weight * value

    aggregated = {
        category: round(weighted_sum.get(category, 0.0) / max(weight_total, 1e-6), 4)
        for category in weighted_sum
    }

    attribution = _build_timestamp_attribution(segments)
    return segments, aggregated, attribution


def blend_temporal_scores(
    base_scores: Dict[str, float],
    temporal_scores: Dict[str, float],
    temporal_weight: float = 0.35,
) -> Dict[str, float]:
    categories = set(base_scores) | set(temporal_scores)
    merged: Dict[str, float] = {}
    for category in categories:
        base = base_scores.get(category, 0.0)
        temporal = temporal_scores.get(category, base)
        merged_value = (1.0 - temporal_weight) * base + temporal_weight * temporal
        merged[category] = round(max(0.0, min(100.0, merged_value)), 4)
    return merged


def _build_timestamp_attribution(segments: List[SegmentScore]) -> List[Dict[str, object]]:
    attributions: List[Dict[str, object]] = []
    for segment in sorted(segments, key=lambda item: item.max_risk_score, reverse=True)[:6]:
        attributions.append(
            {
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "risk_score": segment.max_risk_score,
                "dominant_categories": segment.dominant_categories,
            }
        )
    return attributions


def _build_boundaries(duration: float, window_sec: float, max_segments: int) -> List[Tuple[float, float]]:
    if duration <= 0:
        return [(0.0, window_sec)]

    effective_window = max(1.0, window_sec)
    count = int(duration // effective_window) + 1
    count = max(1, min(count, max_segments))

    boundaries: List[Tuple[float, float]] = []
    for idx in range(count):
        start = idx * effective_window
        end = min(duration, start + effective_window)
        if end <= start:
            end = start + 1.0
        boundaries.append((start, end))
    return boundaries


def _slice_tokens(tokens: List[str], index: int, total_segments: int) -> str:
    if not tokens or total_segments <= 0:
        return ""

    start = int(index * len(tokens) / total_segments)
    end = int((index + 1) * len(tokens) / total_segments)
    return " ".join(tokens[start:end])


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
