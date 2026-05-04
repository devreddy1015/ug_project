from training.v2.risk_engine import (
    compute_final_risk,
    compute_network_diffusion_risk,
    infer_evidence_mode,
)


def test_compute_network_diffusion_risk_high_engagement() -> None:
    engagement = {
        "likes": 55000,
        "shares": 23000,
        "comments": 9000,
        "duets": 1500,
        "stitches": 1200,
        "views": 180000,
        "comment_sentiment": -0.6,
    }
    risk = compute_network_diffusion_risk(
        engagement=engagement,
        safety_harm_score=78.0,
        contradiction_score=64.0,
        evasion_score=51.0,
    )

    assert risk["score"] > 60.0
    assert "cascade_pressure" in risk["components"]


def test_compute_final_risk_combines_all_dimensions() -> None:
    final_risk = compute_final_risk(
        harm_score=80.0,
        viral_harm=70.0,
        contradiction=50.0,
        evasion=30.0,
        cognitive=40.0,
        network_diffusion=60.0,
    )
    assert 60.0 <= final_risk <= 80.0


def test_infer_evidence_mode() -> None:
    assert infer_evidence_mode(80.0) == ("full", False)
    assert infer_evidence_mode(50.0) == ("fallback", False)
    assert infer_evidence_mode(20.0) == ("insufficient", True)
