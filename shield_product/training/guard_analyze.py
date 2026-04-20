import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .guard_multimodal import extract_reel_signals
from .guard_platform import (
    adversarial_evasion_score,
    apply_cultural_adapter,
    compute_network_diffusion_risk,
    cognitive_manipulation_index,
    compute_cross_modal_contradiction,
    compute_viral_harm_potential,
    export_federated_update,
    load_engagement_metadata,
    societal_benefit_details,
    update_creator_profile,
)
from .guard_report import build_guard_report
from .guard_scoring import DualScoringEngine
from .guard_taxonomy import layer_to_categories
from .guard_temporal import blend_temporal_scores, temporal_segment_analysis

if TYPE_CHECKING:
    from .inference import VideoAnalyzer


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
LAYER_MAP = layer_to_categories()


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard multimodal analyzer")
    parser.add_argument("--input", type=str, required=True, help="Video file or folder")
    parser.add_argument("--output-dir", type=str, default="guard_outputs")
    parser.add_argument("--frame-count", type=int, default=6)
    parser.add_argument("--disable-whisper", action="store_true")
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--phase1-model", type=str, default=None)
    parser.add_argument("--phase1-run-summary", type=str, default=None)
    parser.add_argument("--phase2-labels-output", type=str, default=None)
    parser.add_argument("--temporal-window-sec", type=float, default=5.0)
    parser.add_argument("--temporal-recency-bias", type=float, default=1.25)
    parser.add_argument("--temporal-max-segments", type=int, default=18)
    parser.add_argument("--region", type=str, default="auto")
    parser.add_argument("--creator-profile-path", type=str, default=None)
    parser.add_argument("--federated-export-path", type=str, default=None)
    parser.add_argument(
        "--allow-empty-input",
        action="store_true",
        help="Allow completion with zero detected videos (otherwise exits with error).",
    )
    parser.add_argument(
        "--fail-on-item-error",
        action="store_true",
        help="Stop the run on first per-video processing error.",
    )
    args = parser.parse_args()

    run_started_at = _utc_now()
    run_started_perf = time.perf_counter()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = (
        Path(args.creator_profile_path).expanduser().resolve()
        if args.creator_profile_path
        else output_dir / "creator_profiles.json"
    )

    federated_path = (
        Path(args.federated_export_path).expanduser().resolve()
        if args.federated_export_path
        else None
    )

    videos = _collect_videos(input_path)
    if not videos and not args.allow_empty_input:
        raise SystemExit(
            f"No supported video files found for input: {input_path}. "
            "Use --allow-empty-input to emit empty output artifacts."
        )

    scorer = DualScoringEngine()
    phase1 = _build_phase1_analyzer(args.phase1_model, args.phase1_run_summary)

    results: List[Dict[str, object]] = []
    failed_results: List[Dict[str, object]] = []
    phase2_labels: List[Dict[str, object]] = []

    for video_path in videos:
        try:
            report = analyze_single_video_report(
                video_path=video_path,
                output_dir=output_dir,
                scorer=scorer,
                phase1=phase1,
                frame_count=int(args.frame_count),
                disable_whisper=bool(args.disable_whisper),
                disable_ocr=bool(args.disable_ocr),
                temporal_window_sec=float(args.temporal_window_sec),
                temporal_recency_bias=float(args.temporal_recency_bias),
                temporal_max_segments=int(args.temporal_max_segments),
                region_arg=str(args.region),
                profile_path=profile_path,
                federated_path=federated_path,
            )
            results.append(report)

            category_breakdown = report.get("category_breakdown")
            label_map: Dict[str, float] = {}
            if isinstance(category_breakdown, dict):
                label_map = {
                    str(key): round(number / 100.0, 6)
                    for key, value in category_breakdown.items()
                    for number in [_parse_float(value)]
                    if number is not None
                }

            phase2_labels.append(
                {
                    "file": video_path.name,
                    "labels": label_map,
                }
            )
        except Exception as error:
            failed_results.append(_build_failed_result(video_path, error))
            if args.fail_on_item_error:
                raise

    all_results: List[Dict[str, object]] = [*results, *failed_results]
    summary = _build_summary(all_results)
    quality_summary = _build_quality_summary(all_results)
    _write_json(output_dir / "guard_analysis.json", {"results": results, "failed_results": failed_results})
    _write_json(output_dir / "guard_summary.json", summary)
    _write_json(output_dir / "guard_quality_summary.json", quality_summary)
    _write_csv(output_dir / "guard_results.csv", all_results)

    elapsed_seconds = round(time.perf_counter() - run_started_perf, 4)
    run_metadata = {
        "started_at": run_started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total_detected_videos": len(videos),
        "arguments": {
            "frame_count": int(args.frame_count),
            "disable_whisper": bool(args.disable_whisper),
            "disable_ocr": bool(args.disable_ocr),
            "temporal_window_sec": float(args.temporal_window_sec),
            "temporal_recency_bias": float(args.temporal_recency_bias),
            "temporal_max_segments": int(args.temporal_max_segments),
            "region": str(args.region),
            "fail_on_item_error": bool(args.fail_on_item_error),
        },
    }
    _write_json(output_dir / "guard_run_metadata.json", run_metadata)

    if args.phase2_labels_output:
        labels_path = Path(args.phase2_labels_output).expanduser().resolve()
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(labels_path, phase2_labels)

    print("Guard analysis complete")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Completed videos: {summary['completed_videos']}")
    print(f"Failed videos: {summary['failed_videos']}")
    print(f"Safe to watch: {summary['safe_to_watch_count']}")
    print(f"Average safety score: {summary['average_safety_score']}")
    print(f"Average viral harm potential: {summary['average_viral_harm_potential']}")
    print(f"Average network diffusion risk: {summary['average_network_diffusion_risk']}")
    print(f"Average analysis confidence: {summary['average_analysis_confidence']}")
    print(f"Insufficient evidence videos: {summary['insufficient_evidence_count']}")

    insufficient_count = int(summary.get("insufficient_evidence_count", 0))
    if insufficient_count > 0:
        print(
            "Warning: low-confidence moderation results detected. "
            "Consider enabling transcription/OCR and providing caption/metadata sidecars."
        )
        missing_components = _collect_missing_components(all_results)
        if missing_components:
            print(
                "Missing extractor components detected: "
                + ", ".join(missing_components)
            )

    print(f"Output directory: {output_dir}")


def _collect_videos(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return [input_path]

    if not input_path.exists():
        return []

    videos = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def _build_phase1_analyzer(
    model_path: Optional[str], run_summary_path: Optional[str]
) -> Optional["VideoAnalyzer"]:
    if not model_path:
        return None

    from .inference import InferenceConfig, VideoAnalyzer

    resolved_model = Path(model_path).expanduser().resolve()
    resolved_summary = (
        Path(run_summary_path).expanduser().resolve() if run_summary_path else None
    )

    if not resolved_model.exists():
        return None

    # Prefer no-frame sibling model when available for faster phase-1 execution.
    summary_payload = _read_json(resolved_summary) if resolved_summary else {}
    use_frames = bool(
        summary_payload.get("feature_spec", {}).get("use_frames", False)
        if isinstance(summary_payload, dict)
        else False
    )
    if use_frames:
        sibling_dir = resolved_model.parent.parent / resolved_model.parent.name.replace("_frames", "")
        sibling_model = sibling_dir / resolved_model.name
        sibling_summary = sibling_dir / "run_summary.json"
        if sibling_model.exists() and sibling_summary.exists():
            resolved_model = sibling_model
            resolved_summary = sibling_summary

    config = InferenceConfig(
        model_path=resolved_model,
        run_summary_path=resolved_summary,
        threshold=50.0,
        preview_frames=0,
        save_previews=False,
    )
    return VideoAnalyzer(config)


def _phase1_harm_probability(
    analyzer: "VideoAnalyzer", video_path: Path, output_dir: Path
) -> float:
    try:
        result = analyzer.analyze_video(video_path, output_dir)
        return float(result.get("harm_score_pct", 0.0))
    except Exception:
        return 0.0


def analyze_single_video_report(
    *,
    video_path: Path,
    output_dir: Path,
    scorer: DualScoringEngine,
    phase1: Optional["VideoAnalyzer"],
    frame_count: int,
    disable_whisper: bool,
    disable_ocr: bool,
    temporal_window_sec: float,
    temporal_recency_bias: float,
    temporal_max_segments: int,
    region_arg: str,
    profile_path: Path,
    federated_path: Optional[Path],
    display_video_path: Optional[str] = None,
) -> Dict[str, object]:
    signals = extract_reel_signals(
        video_path,
        frame_count=int(frame_count),
        enable_whisper=not disable_whisper,
        enable_ocr=not disable_ocr,
    )
    _validate_media_signals(signals.metadata)

    score_result = scorer.score(signals.combined_text)

    segment_scores, temporal_scores, timestamp_attr = temporal_segment_analysis(
        video_path=video_path,
        transcript_text=signals.transcript_text,
        caption_text=signals.caption_text,
        hashtags=signals.hashtags,
        vector_scorer=scorer.score_vector_only,
        window_sec=float(temporal_window_sec),
        recency_bias=float(temporal_recency_bias),
        max_segments=int(temporal_max_segments),
    )

    combined_scores = blend_temporal_scores(
        score_result.combined_scores,
        temporal_scores,
        temporal_weight=0.35,
    )

    phase1_probability = (
        _phase1_harm_probability(phase1, video_path, output_dir)
        if phase1 is not None
        else 0.0
    )
    combined_scores = _inject_phase1_prior(combined_scores, phase1_probability)

    engagement = load_engagement_metadata(video_path)
    region = _resolve_region(region_arg, engagement)
    combined_scores = apply_cultural_adapter(combined_scores, region)

    harm_avg = _average_categories(combined_scores, LAYER_MAP.get("Safety & Harm Layer", []))
    viral_harm = compute_viral_harm_potential(harm_avg, engagement)
    contradiction = compute_cross_modal_contradiction(
        signals.transcript_text,
        signals.ocr_text,
        signals.caption_text,
        signals.frame_signals,
    )
    evasion = adversarial_evasion_score(signals.combined_text)
    cognitive_index = cognitive_manipulation_index(
        signals.metadata,
        signals.frame_signals,
        engagement,
    )
    sbi_details = societal_benefit_details(combined_scores, cognitive_index)
    network_diffusion = compute_network_diffusion_risk(
        engagement=engagement,
        safety_harm_score=harm_avg,
        contradiction_score=contradiction,
        evasion_score=evasion,
    )

    report_video_path = str(display_video_path).strip() if display_video_path else str(video_path)
    if not report_video_path:
        report_video_path = str(video_path)

    report = build_guard_report(
        video_path=report_video_path,
        combined_scores=combined_scores,
        metadata={
            **signals.metadata,
            "frame_signals": signals.frame_signals,
            "hashtags": signals.hashtags,
            "caption_text": signals.caption_text,
            "engagement": engagement,
            "region": region,
        },
        llm_verdict=score_result.llm_verdict,
        phase1_harm_probability=phase1_probability,
    )

    report["engine_details"] = {
        "used_groq": score_result.used_groq,
        "used_embeddings": score_result.used_embeddings,
        "used_chromadb": score_result.used_chromadb,
        "llm_error": getattr(score_result, "llm_error", ""),
    }
    report["llm_scores"] = score_result.llm_scores
    report["vector_scores"] = score_result.vector_scores
    report["temporal_segments"] = [
        {
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "max_risk_score": segment.max_risk_score,
            "dominant_categories": segment.dominant_categories,
            "category_scores": segment.category_scores,
        }
        for segment in segment_scores
    ]
    report["timestamp_attribution"] = timestamp_attr
    report["viral_harm_potential"] = viral_harm
    report["cross_modal_contradiction_score"] = contradiction
    report["adversarial_evasion_score"] = evasion
    report["cognitive_manipulation_index"] = cognitive_index
    report["societal_benefit_index"] = sbi_details["sbi_score"]
    report["sbi_details"] = sbi_details
    report["network_diffusion_risk"] = network_diffusion["score"]
    report["network_diffusion_details"] = network_diffusion["components"]

    # Reconcile risk with post-scoring behavioral signals so the headline metric
    # reflects contradiction/evasion/diffusion, not only taxonomy averages.
    try:
        base_risk = float(report.get("overall_risk_score_out_of_100", 0.0))
    except (TypeError, ValueError):
        base_risk = 0.0
    if base_risk <= 0.0:
        try:
            base_safety = float(report.get("overall_safety_score_out_of_100", 50.0))
        except (TypeError, ValueError):
            base_safety = 50.0
        base_risk = max(0.0, min(100.0, 100.0 - base_safety))

    behavioral_risk = (
        0.45 * float(contradiction)
        + 0.20 * float(evasion)
        + 0.15 * float(viral_harm)
        + 0.10 * float(network_diffusion.get("score", 0.0))
        + 0.10 * float(phase1_probability)
    )
    holistic_risk = max(base_risk, min(100.0, behavioral_risk))
    report["overall_risk_score_out_of_100"] = round(holistic_risk, 4)
    report["overall_safety_score_out_of_100"] = round(max(0.0, 100.0 - holistic_risk), 4)

    warning_notes = list(report.get("warning_notes", []))
    if contradiction >= 70.0:
        warning_notes.append(f"cross_modal_contradiction: {contradiction:.1f}")
    if evasion >= 70.0:
        warning_notes.append(f"adversarial_evasion: {evasion:.1f}")
    if float(network_diffusion.get("score", 0.0)) >= 70.0:
        warning_notes.append(
            f"network_diffusion_risk: {float(network_diffusion.get('score', 0.0)):.1f}"
        )
    report["warning_notes"] = warning_notes[:8]

    report["content_age_rating"] = _upgrade_age_rating(
        str(report.get("content_age_rating", "U")),
        viral_harm,
        contradiction,
        evasion,
    )
    if (
        viral_harm >= 65.0
        or contradiction >= 70.0
        or evasion >= 70.0
        or float(network_diffusion.get("score", 0.0)) >= 70.0
        or float(report.get("overall_risk_score_out_of_100", 0.0)) >= 55.0
    ):
        report["safe_to_watch"] = False

    creator_id = _resolve_creator_id(Path(report_video_path), engagement)
    creator_profile = update_creator_profile(
        profile_path=profile_path,
        creator_id=creator_id,
        video_path=report_video_path,
        safety_score=float(report.get("overall_safety_score_out_of_100", 0.0)),
        harm_probability=float(report.get("phase1_harm_probability", 0.0)),
    )
    report["creator_id"] = creator_id
    report["creator_profile"] = creator_profile

    if federated_path is not None:
        export_federated_update(
            output_path=federated_path,
            creator_id=creator_id,
            video_path=report_video_path,
            category_breakdown=report.get("category_breakdown", {}),
            phase1_harm_probability=phase1_probability,
        )

    report["processing_status"] = "completed"
    report["error_message"] = None
    return report


def _build_summary(results: List[Dict[str, object]]) -> Dict[str, object]:
    if not results:
        return {
            "total_videos": 0,
            "completed_videos": 0,
            "failed_videos": 0,
            "safe_to_watch_count": 0,
            "average_safety_score": 0.0,
            "average_good_for_society": 0.0,
            "average_viral_harm_potential": 0.0,
            "average_network_diffusion_risk": 0.0,
            "average_sbi": 0.0,
            "average_analysis_confidence": 0.0,
            "insufficient_evidence_count": 0,
            "age_rating_distribution": {},
        }

    completed_results = [
        entry
        for entry in results
        if str(entry.get("processing_status", "completed")) == "completed"
    ]
    failed_videos = len(results) - len(completed_results)

    if not completed_results:
        return {
            "total_videos": len(results),
            "completed_videos": 0,
            "failed_videos": failed_videos,
            "safe_to_watch_count": 0,
            "average_safety_score": 0.0,
            "average_good_for_society": 0.0,
            "average_viral_harm_potential": 0.0,
            "average_network_diffusion_risk": 0.0,
            "average_sbi": 0.0,
            "average_analysis_confidence": 0.0,
            "insufficient_evidence_count": 0,
            "age_rating_distribution": {},
        }

    safe_to_watch_count = 0
    safety_total = 0.0
    society_total = 0.0
    viral_total = 0.0
    network_diffusion_total = 0.0
    sbi_total = 0.0
    confidence_total = 0.0
    insufficient_evidence_count = 0
    age_distribution: Dict[str, int] = {}

    for entry in completed_results:
        if entry.get("safe_to_watch"):
            safe_to_watch_count += 1

        safety_total += float(entry.get("overall_safety_score_out_of_100", 0.0))
        society_total += float(entry.get("good_for_society_percentage", 0.0))
        viral_total += float(entry.get("viral_harm_potential", 0.0))
        network_diffusion_total += float(entry.get("network_diffusion_risk", 0.0))
        sbi_total += float(entry.get("societal_benefit_index", 0.0))
        confidence_total += float(entry.get("analysis_confidence", 0.0))
        if bool(entry.get("insufficient_evidence", False)):
            insufficient_evidence_count += 1

        age = str(entry.get("content_age_rating", "U"))
        age_distribution[age] = age_distribution.get(age, 0) + 1

    total = len(completed_results)

    return {
        "total_videos": len(results),
        "completed_videos": len(completed_results),
        "failed_videos": failed_videos,
        "safe_to_watch_count": safe_to_watch_count,
        "average_safety_score": round(safety_total / total, 4),
        "average_good_for_society": round(society_total / total, 4),
        "average_viral_harm_potential": round(viral_total / total, 4),
        "average_network_diffusion_risk": round(network_diffusion_total / total, 4),
        "average_sbi": round(sbi_total / total, 4),
        "average_analysis_confidence": round(confidence_total / total, 4),
        "insufficient_evidence_count": insufficient_evidence_count,
        "age_rating_distribution": age_distribution,
    }


def _build_quality_summary(results: List[Dict[str, object]]) -> Dict[str, object]:
    if not results:
        return {
            "total_videos": 0,
            "completed_videos": 0,
            "failed_videos": 0,
            "success_rate_percentage": 0.0,
            "high_risk_count": 0,
            "insufficient_evidence_count": 0,
            "average_analysis_confidence": 0.0,
            "average_network_diffusion_risk": 0.0,
        }

    completed_results = [
        entry
        for entry in results
        if str(entry.get("processing_status", "completed")) == "completed"
    ]
    total = len(results)
    completed = len(completed_results)
    failed = total - completed

    insufficient_count = sum(
        1 for entry in completed_results if bool(entry.get("insufficient_evidence", False))
    )
    high_risk_count = sum(
        1
        for entry in completed_results
        if (
            float(entry.get("viral_harm_potential", 0.0)) >= 65.0
            or float(entry.get("network_diffusion_risk", 0.0)) >= 70.0
            or float(entry.get("overall_safety_score_out_of_100", 0.0)) <= 40.0
            or str(entry.get("content_age_rating", "U")) == "18+"
        )
    )

    avg_conf = 0.0
    avg_network_diffusion = 0.0
    if completed_results:
        avg_conf = round(
            sum(float(entry.get("analysis_confidence", 0.0)) for entry in completed_results)
            / len(completed_results),
            4,
        )
        avg_network_diffusion = round(
            sum(float(entry.get("network_diffusion_risk", 0.0)) for entry in completed_results)
            / len(completed_results),
            4,
        )

    return {
        "total_videos": total,
        "completed_videos": completed,
        "failed_videos": failed,
        "success_rate_percentage": round((completed / total) * 100.0, 4),
        "high_risk_count": high_risk_count,
        "insufficient_evidence_count": insufficient_count,
        "average_analysis_confidence": avg_conf,
        "average_network_diffusion_risk": avg_network_diffusion,
    }


def _write_json(path: Path, payload: Dict[str, object] | List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "processing_status",
        "error_message",
        "video_path",
        "overall_safety_score_out_of_100",
        "overall_risk_score_out_of_100",
        "good_for_society_percentage",
        "societal_benefit_index",
        "viral_harm_potential",
        "network_diffusion_risk",
        "safe_to_watch",
        "content_age_rating",
        "phase1_harm_probability",
        "cross_modal_contradiction_score",
        "adversarial_evasion_score",
        "cognitive_manipulation_index",
        "analysis_confidence",
        "insufficient_evidence",
        "evidence_mode",
        "creator_id",
        "verdict",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "processing_status": row.get("processing_status", "completed"),
                    "error_message": row.get("error_message"),
                    "video_path": row.get("video_path"),
                    "overall_safety_score_out_of_100": row.get("overall_safety_score_out_of_100"),
                    "overall_risk_score_out_of_100": row.get("overall_risk_score_out_of_100"),
                    "good_for_society_percentage": row.get("good_for_society_percentage"),
                    "societal_benefit_index": row.get("societal_benefit_index"),
                    "viral_harm_potential": row.get("viral_harm_potential"),
                    "network_diffusion_risk": row.get("network_diffusion_risk"),
                    "safe_to_watch": row.get("safe_to_watch"),
                    "content_age_rating": row.get("content_age_rating"),
                    "phase1_harm_probability": row.get("phase1_harm_probability"),
                    "cross_modal_contradiction_score": row.get("cross_modal_contradiction_score"),
                    "adversarial_evasion_score": row.get("adversarial_evasion_score"),
                    "cognitive_manipulation_index": row.get("cognitive_manipulation_index"),
                    "analysis_confidence": row.get("analysis_confidence"),
                    "insufficient_evidence": row.get("insufficient_evidence"),
                    "evidence_mode": row.get("evidence_mode"),
                    "creator_id": row.get("creator_id"),
                    "verdict": row.get("verdict"),
                }
            )


def _inject_phase1_prior(
    combined_scores: Dict[str, float], phase1_probability: float
) -> Dict[str, float]:
    if phase1_probability <= 0.0:
        return dict(combined_scores)

    adjusted = dict(combined_scores)
    harm_categories = LAYER_MAP.get("Safety & Harm Layer", [])
    societal_categories = LAYER_MAP.get("Societal Impact Layer", [])

    for category in harm_categories:
        prior = phase1_probability * 0.70
        adjusted[category] = max(adjusted.get(category, 0.0), round(prior, 4))

    for category in societal_categories:
        prior = phase1_probability * 0.35
        adjusted[category] = max(adjusted.get(category, 0.0), round(prior, 4))

    return adjusted


def _resolve_region(region_arg: str, engagement: Dict[str, float | str]) -> str:
    if region_arg and region_arg.lower() != "auto":
        return region_arg.lower()
    region = str(engagement.get("region", "global"))
    return region.lower() if region else "global"


def _resolve_creator_id(video_path: Path, engagement: Dict[str, float | str]) -> str:
    creator = str(engagement.get("creator_id", "")).strip()
    if creator and creator != "unknown_creator":
        return creator
    return f"creator_{video_path.stem}"


def _average_categories(scores: Dict[str, float], categories: List[str]) -> float:
    if not categories:
        return 0.0
    values = [float(scores.get(category, 0.0)) for category in categories]
    return sum(values) / len(values)


def _upgrade_age_rating(
    current_rating: str,
    viral_harm: float,
    contradiction: float,
    evasion: float,
) -> str:
    order = {"U": 0, "7+": 1, "13+": 2, "18+": 3}
    current = order.get(current_rating, 0)

    target = current
    if viral_harm >= 80.0 or contradiction >= 80.0 or evasion >= 80.0:
        target = max(target, order["18+"])
    elif viral_harm >= 60.0 or contradiction >= 65.0 or evasion >= 65.0:
        target = max(target, order["13+"])
    elif viral_harm >= 35.0 or contradiction >= 45.0 or evasion >= 45.0:
        target = max(target, order["7+"])

    inverse = {value: key for key, value in order.items()}
    return inverse.get(target, current_rating)


def _read_json(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return {}
    return {}


def _build_failed_result(video_path: Path, error: Exception) -> Dict[str, object]:
    message = str(error).strip() or "Unknown processing failure"
    if len(message) > 500:
        message = message[:500] + "..."

    return {
        "processing_status": "failed",
        "error_message": message,
        "video_path": str(video_path),
        "overall_safety_score_out_of_100": 0.0,
        "good_for_society_percentage": 0.0,
        "societal_benefit_index": 0.0,
        "viral_harm_potential": 0.0,
        "network_diffusion_risk": 0.0,
        "safe_to_watch": False,
        "content_age_rating": "U",
        "phase1_harm_probability": 0.0,
        "cross_modal_contradiction_score": 0.0,
        "adversarial_evasion_score": 0.0,
        "cognitive_manipulation_index": 0.0,
        "analysis_confidence": 0.0,
        "insufficient_evidence": True,
        "evidence_mode": "failed",
        "creator_id": None,
        "verdict": f"Analysis failed: {message}",
    }


def _validate_media_signals(metadata: Dict[str, object]) -> None:
    diagnostics = metadata.get("extraction_diagnostics", {})
    ffprobe_available = True
    if isinstance(diagnostics, dict):
        available = diagnostics.get("available", {})
        if isinstance(available, dict):
            ffprobe_available = bool(available.get("ffprobe", True))

    if not ffprobe_available:
        return

    duration = 0.0
    try:
        duration = float(metadata.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    has_video = metadata.get("has_video")
    if has_video is False:
        raise ValueError("Media integrity validation failed: no video stream detected")
    if duration <= 0.0:
        raise ValueError("Media integrity validation failed: non-positive video duration")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_missing_components(results: List[Dict[str, object]]) -> List[str]:
    missing: set[str] = set()
    for entry in results:
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            continue
        diagnostics = metadata.get("extraction_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        values = diagnostics.get("missing_components", [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                missing.add(value.strip())
    return sorted(missing)


def _parse_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
