from pathlib import Path

from training.v2.config import GuardV2Config
from training.v2.pipeline import GuardV2Pipeline
from training.v2.types import MediaAsset, SemanticResult, SignalPayload


class StubExtractor:
    def extract(self, video_path: Path, *, frame_count: int, enable_whisper: bool, enable_ocr: bool) -> SignalPayload:
        return SignalPayload(
            transcript_text="we must attack now share now",
            ocr_text="danger",
            caption_text="urgent warning",
            hashtags=["#trend", "#signal"],
            frame_signals={"avg_motion": 65.0, "avg_brightness": 85.0, "cuts_per_minute": 72.0},
            metadata={
                "duration": 44.0,
                "watch_ratio": 0.74,
                "extraction_diagnostics": {
                    "available": {
                        "ffprobe": True,
                        "whisper": True,
                        "ocr": True,
                    }
                },
            },
        )


class StubScorer:
    def score(self, combined_text: str) -> SemanticResult:
        return SemanticResult(
            combined_scores={"hate_speech": 82.0, "violence": 71.0, "neutral": 12.0},
            llm_verdict="High risk",
            vector_scores={"hate_speech": 80.0},
            llm_scores={"violence": 72.0},
            engine_details={"used_groq": True, "used_embeddings": True, "used_chromadb": True},
        )


class StubEngagementProvider:
    def load(self, video_path: Path):
        return {
            "likes": 12000,
            "shares": 6400,
            "comments": 2400,
            "duets": 300,
            "stitches": 200,
            "views": 75000,
            "comment_sentiment": -0.4,
            "region": "us",
        }


class IdentityCulturalAdapter:
    def adapt(self, category_scores, region: str):
        return dict(category_scores)


class FailingExtractor:
    def extract(self, video_path: Path, *, frame_count: int, enable_whisper: bool, enable_ocr: bool):
        raise RuntimeError("boom")


def test_pipeline_analyze_asset() -> None:
    pipeline = GuardV2Pipeline(
        config=GuardV2Config(),
        signal_extractor=StubExtractor(),
        semantic_scorer=StubScorer(),
        engagement_provider=StubEngagementProvider(),
        cultural_adapter=IdentityCulturalAdapter(),
    )

    report = pipeline.analyze_asset(MediaAsset(path=Path("sample.mp4"), source_id="video_sample"))

    assert report["processing_status"] == "completed"
    assert report["region"] == "us"
    assert report["overall_risk_score_out_of_100"] > 0.0
    assert report["network_diffusion_risk"] > 0.0
    assert isinstance(report["top_risk_categories"], list)
    assert report["evidence_mode"] in {"full", "fallback", "insufficient"}


def test_pipeline_collects_failures() -> None:
    pipeline = GuardV2Pipeline(
        config=GuardV2Config(),
        signal_extractor=FailingExtractor(),
        semantic_scorer=StubScorer(),
        engagement_provider=StubEngagementProvider(),
        cultural_adapter=IdentityCulturalAdapter(),
    )

    completed, failed = pipeline.analyze_assets([
        MediaAsset(path=Path("broken.mp4"), source_id="video_broken")
    ])

    assert completed == []
    assert len(failed) == 1
    assert failed[0]["processing_status"] == "failed"
