from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .features import FeatureSpec, build_feature_spec, extract_features
from .labels import find_labels_for_path, infer_binary_label, load_label_map


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md"}


@dataclass(frozen=True)
class TrainingItem:
    path: Path
    modality: str
    label: torch.Tensor


class TrainingDataset(Dataset):
    def __init__(
        self,
        items: Sequence[TrainingItem],
        feature_spec: FeatureSpec,
    ) -> None:
        self._feature_spec = feature_spec
        self._items = list(items)
        self._features: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

        for item in self._items:
            features = extract_features(item.path, feature_spec)
            self._features.append(torch.tensor(features, dtype=torch.float32))
            self._labels.append(item.label)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._features[index], self._labels[index]


def collect_items(
    dataset_dir: Path,
    modality: str,
    multi_label: bool,
    labels_path: Optional[Path],
    max_files: Optional[int] = None,
    frame_count: int = 4,
    use_frames: bool = True,
) -> Tuple[List[TrainingItem], FeatureSpec, List[str]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    label_map: Dict[str, Dict[str, float]] = {}
    categories: List[str] = []

    if multi_label:
        if labels_path is None:
            raise ValueError("labels_path is required for multi-label training")
        label_map, categories = load_label_map(labels_path)

    feature_spec = build_feature_spec(modality, frame_count, use_frames)

    items: List[TrainingItem] = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if modality == "video" and path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if modality == "audio" and path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if modality == "image" and path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if modality == "text" and path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        label = _build_label(path, dataset_dir, multi_label, label_map, categories)
        if label is None:
            continue
        items.append(TrainingItem(path=path, modality=modality, label=label))

        if max_files is not None and len(items) >= max_files:
            break

    return items, feature_spec, categories


def _build_label(
    path: Path,
    dataset_dir: Path,
    multi_label: bool,
    label_map: Dict[str, Dict[str, float]],
    categories: List[str],
) -> Optional[torch.Tensor]:
    if multi_label:
        labels = find_labels_for_path(path, dataset_dir, label_map)
        if labels is None:
            return None
        vector = [float(labels.get(category, 0.0)) for category in categories]
        return torch.tensor(vector, dtype=torch.float32)

    binary = infer_binary_label(path)
    if binary is None:
        return None
    return torch.tensor([float(binary)], dtype=torch.float32)
