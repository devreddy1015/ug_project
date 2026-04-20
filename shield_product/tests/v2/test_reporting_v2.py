import json
from pathlib import Path

from training.v2.reporting import build_quality_summary, build_summary, write_outputs


def _completed_row() -> dict:
    return {
        "processing_status": "completed",
        "error_message": None,
        "video_path": "a.mp4",
        "source_id": "video_a",
        "region": "global",
        "overall_risk_score_out_of_100": 62.0,
        "overall_safety_score_out_of_100": 38.0,
        "good_for_society_percentage": 42.0,
        "viral_harm_potential": 66.0,
        "network_diffusion_risk": 58.0,
        "safe_to_watch": False,
        "content_age_rating": "16+",
        "phase1_harm_probability": 0.0,
        "cross_modal_contradiction_score": 32.0,
        "adversarial_evasion_score": 20.0,
        "cognitive_manipulation_index": 44.0,
        "analysis_confidence": 74.0,
        "insufficient_evidence": False,
        "evidence_mode": "full",
        "verdict": "High risk",
    }


def _failed_row() -> dict:
    return {
        "processing_status": "failed",
        "error_message": "failure",
        "video_path": "b.mp4",
        "source_id": "video_b",
        "region": "unknown",
        "overall_risk_score_out_of_100": 50.0,
        "overall_safety_score_out_of_100": 50.0,
        "good_for_society_percentage": 50.0,
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
        "verdict": "failed",
    }


def test_build_summary_and_quality_summary() -> None:
    completed = [_completed_row()]
    failed = [_failed_row()]

    summary = build_summary(completed, failed)
    quality = build_quality_summary(completed, failed)

    assert summary["total_videos"] == 2
    assert summary["failed_videos"] == 1
    assert quality["success_rate_percentage"] == 50.0


def test_write_outputs(tmp_path: Path) -> None:
    completed = [_completed_row()]
    failed = [_failed_row()]

    write_outputs(
        output_dir=tmp_path,
        completed_results=completed,
        failed_results=failed,
        run_metadata={"run": "ok"},
    )

    expected = [
        "guard_analysis.json",
        "guard_summary.json",
        "guard_quality_summary.json",
        "guard_results.csv",
        "guard_run_metadata.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), f"missing {name}"

    payload = json.loads((tmp_path / "guard_analysis.json").read_text(encoding="utf-8"))
    assert len(payload["results"]) == 1
    assert len(payload["failed_results"]) == 1
