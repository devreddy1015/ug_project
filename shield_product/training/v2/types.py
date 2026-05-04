from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class MediaAsset:
    path: Path
    source_id: str


@dataclass
class SignalPayload:
    transcript_text: str = ""
    ocr_text: str = ""
    caption_text: str = ""
    hashtags: List[str] = field(default_factory=list)
    frame_signals: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def combined_text(self) -> str:
        hashtag_blob = " ".join(self.hashtags)
        return " ".join([self.transcript_text, self.ocr_text, self.caption_text, hashtag_blob]).strip()


@dataclass
class SemanticResult:
    combined_scores: Dict[str, float]
    llm_verdict: str = ""
    vector_scores: Dict[str, float] = field(default_factory=dict)
    llm_scores: Dict[str, float] = field(default_factory=dict)
    engine_details: Dict[str, object] = field(default_factory=dict)
