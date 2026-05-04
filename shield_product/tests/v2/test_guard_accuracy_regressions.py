from training.guard_report import _evidence_quality, build_guard_report
from training.guard_scoring import _similarity_to_score


def test_fallback_only_with_no_primary_text_is_insufficient() -> None:
    combined_scores = {"hate_speech": 32.0, "violence": 18.0, "misinformation": 24.0}
    metadata = {
        "extraction_diagnostics": {
            "fallback_context_applied": True,
            "source_presence": {
                "has_transcript": False,
                "has_ocr_text": False,
                "has_caption_text": False,
                "has_hashtags": False,
            },
            "missing_components": ["whisper", "pytesseract"],
        }
    }

    evidence = _evidence_quality(combined_scores, phase1_harm_probability=0.0, metadata=metadata)

    assert evidence["fallback_only"] is True
    assert evidence["insufficient"] is True
    assert float(evidence["confidence"]) <= 0.20


def test_fallback_only_allows_non_insufficient_with_strong_phase1() -> None:
    combined_scores = {"hate_speech": 45.0, "violence": 41.0, "misinformation": 20.0}
    metadata = {
        "extraction_diagnostics": {
            "fallback_context_applied": True,
            "source_presence": {
                "has_transcript": False,
                "has_ocr_text": False,
                "has_caption_text": False,
                "has_hashtags": False,
            },
            "missing_components": ["whisper"],
        }
    }

    evidence = _evidence_quality(combined_scores, phase1_harm_probability=85.0, metadata=metadata)

    assert evidence["fallback_only"] is True
    assert evidence["insufficient"] is False


def test_similarity_to_score_is_calibrated_for_weak_matches() -> None:
    assert _similarity_to_score(0.05) == 0.0
    assert _similarity_to_score(0.20) < 10.0
    assert 25.0 <= _similarity_to_score(0.50) <= 60.0
    assert _similarity_to_score(0.90) >= 90.0


def test_build_guard_report_includes_risk_field() -> None:
    report = build_guard_report(
        video_path="sample.mp4",
        combined_scores={"hate_speech": 30.0, "violence": 15.0, "educational_value": 10.0},
        metadata={"extraction_diagnostics": {"source_presence": {"has_transcript": True}}},
        llm_verdict="",
        phase1_harm_probability=5.0,
    )

    safety = float(report["overall_safety_score_out_of_100"])
    risk = float(report["overall_risk_score_out_of_100"])
    assert abs((100.0 - safety) - risk) < 1e-6


def test_evidence_quality_prefers_rich_primary_signal() -> None:
    combined_scores = {"hate_speech": 2.0, "violence": 1.0, "misinformation": 1.5}
    metadata = {
        "extraction_diagnostics": {
            "fallback_context_applied": False,
            "source_presence": {
                "has_transcript": True,
                "has_ocr_text": True,
                "has_caption_text": True,
                "has_hashtags": True,
            },
            "text_lengths": {
                "combined_chars": 3200,
            },
            "available": {
                "opencv": True,
                "ffmpeg": True,
                "ffprobe": True,
                "whisper": True,
                "pytesseract": True,
                "tesseract": True,
            },
            "missing_components": [],
        }
    }

    evidence = _evidence_quality(combined_scores, phase1_harm_probability=0.0, metadata=metadata)

    assert evidence["insufficient"] is False
    assert float(evidence["confidence"]) >= 0.75
