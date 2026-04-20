from dataclasses import dataclass


@dataclass(frozen=True)
class GuardV2Config:
    frame_count: int = 8
    enable_whisper: bool = True
    enable_ocr: bool = True
    region: str = "auto"
    final_block_threshold: float = 70.0
    network_block_threshold: float = 70.0
    caution_threshold: float = 45.0
    top_categories: int = 6
