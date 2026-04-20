from pathlib import Path
from typing import List, Optional

from ..DataObjects.AudioDataObject import AudioDataObject
from ..DataObjects.ImageDataObject import ImageDataObject
from ..DataObjects.TextDataObject import TextDataObject
from ..DataObjects.VideoDataObject import VideoDataObject
from ..DataObjects.Base import BaseDataObject


TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _detect_label(path: Path) -> Optional[int]:
    path_lower = str(path).lower()
    if "non_hate" in path_lower or "non-hate" in path_lower:
        return 0
    if "hate" in path_lower:
        return 1
    return None


def segregate_files(root: Path, max_files: Optional[int] = None) -> List[BaseDataObject]:
    results: List[BaseDataObject] = []
    if not root.exists():
        return results

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        label = _detect_label(path)

        if suffix in TEXT_EXTENSIONS:
            results.append(TextDataObject(path=path, modality="text", label=label))
        elif suffix in IMAGE_EXTENSIONS:
            results.append(ImageDataObject(path=path, modality="image", label=label))
        elif suffix in AUDIO_EXTENSIONS:
            results.append(AudioDataObject(path=path, modality="audio", label=label))
        elif suffix in VIDEO_EXTENSIONS:
            results.append(VideoDataObject(path=path, modality="video", label=label))

        if max_files is not None and len(results) >= max_files:
            break

    return results
