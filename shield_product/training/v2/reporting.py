import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def build_summary(
    completed_results: List[Dict[str, object]],
    failed_results: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    failed = failed_results or []
    if not completed_results:
        return {
            "total_videos": len(failed),
            "completed_videos": 0,
            "failed_videos": len(failed),
            "safe_to_watch_count": 0,
            "average_safety_score": 0.0,
            "average_risk_score": 0.0,
            "average_viral_harm_potential": 0.0,
            "average_network_diffusion_risk": 0.0,
            "average_analysis_confidence": 0.0,
            "insufficient_evidence_count": len(failed),
            "age_rating_distribution": {},
        }

    safe_count = 0
    safety_total = 0.0
    risk_total = 0.0
    viral_total = 0.0
    network_total = 0.0
    confidence_total = 0.0
    insufficient_count = 0
    age_distribution: Dict[str, int] = {}

    for row in completed_results:
        if bool(row.get("safe_to_watch")):
            safe_count += 1

        safety_total += float(row.get("overall_safety_score_out_of_100", 0.0))
        risk_total += float(row.get("overall_risk_score_out_of_100", 0.0))
        viral_total += float(row.get("viral_harm_potential", 0.0))
        network_total += float(row.get("network_diffusion_risk", 0.0))
        confidence_total += float(row.get("analysis_confidence", 0.0))

        if bool(row.get("insufficient_evidence", False)):
            insufficient_count += 1

        age = str(row.get("content_age_rating", "U"))
        age_distribution[age] = age_distribution.get(age, 0) + 1

    total = len(completed_results)
    return {
        "total_videos": len(completed_results) + len(failed),
        "completed_videos": len(completed_results),
        "failed_videos": len(failed),
        "safe_to_watch_count": safe_count,
        "average_safety_score": round(safety_total / total, 4),
        "average_risk_score": round(risk_total / total, 4),
        "average_viral_harm_potential": round(viral_total / total, 4),
        "average_network_diffusion_risk": round(network_total / total, 4),
        "average_analysis_confidence": round(confidence_total / total, 4),
        "insufficient_evidence_count": insufficient_count,
        "age_rating_distribution": age_distribution,
    }


def build_quality_summary(
    completed_results: List[Dict[str, object]],
    failed_results: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    failed = failed_results or []
    total = len(completed_results) + len(failed)
    if total == 0:
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

    high_risk_count = sum(
        1
        for row in completed_results
        if (
            float(row.get("overall_risk_score_out_of_100", 0.0)) >= 70.0
            or float(row.get("network_diffusion_risk", 0.0)) >= 70.0
            or str(row.get("content_age_rating", "U")) == "18+"
        )
    )

    insufficient_count = sum(
        1 for row in completed_results if bool(row.get("insufficient_evidence", False))
    ) + len(failed)

    avg_conf = 0.0
    avg_network = 0.0
    if completed_results:
        avg_conf = round(
            sum(float(row.get("analysis_confidence", 0.0)) for row in completed_results)
            / len(completed_results),
            4,
        )
        avg_network = round(
            sum(float(row.get("network_diffusion_risk", 0.0)) for row in completed_results)
            / len(completed_results),
            4,
        )

    return {
        "total_videos": total,
        "completed_videos": len(completed_results),
        "failed_videos": len(failed),
        "success_rate_percentage": round((len(completed_results) / total) * 100.0, 4),
        "high_risk_count": high_risk_count,
        "insufficient_evidence_count": insufficient_count,
        "average_analysis_confidence": avg_conf,
        "average_network_diffusion_risk": avg_network,
    }


def write_outputs(
    output_dir: Path,
    completed_results: List[Dict[str, object]],
    failed_results: List[Dict[str, object]],
    run_metadata: Optional[Dict[str, object]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(completed_results, failed_results)
    quality_summary = build_quality_summary(completed_results, failed_results)
    all_rows = [*completed_results, *failed_results]

    _write_json(output_dir / "guard_analysis.json", {"results": completed_results, "failed_results": failed_results})
    _write_json(output_dir / "guard_summary.json", summary)
    _write_json(output_dir / "guard_quality_summary.json", quality_summary)
    _write_csv(output_dir / "guard_results.csv", all_rows)
    _write_json(output_dir / "guard_run_metadata.json", run_metadata or {})


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "processing_status",
        "error_message",
        "video_path",
        "source_id",
        "region",
        "overall_risk_score_out_of_100",
        "overall_safety_score_out_of_100",
        "good_for_society_percentage",
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
        "verdict",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
