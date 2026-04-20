from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: Path
    output_dir: Path
    modality: str
    labels_path: Optional[Path] = None
    multi_label: bool = False
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    validation_split: float = 0.2
    seed: int = 42
    max_files: Optional[int] = None
    frame_count: int = 4
    use_frames: bool = True
    threshold: float = 0.5
