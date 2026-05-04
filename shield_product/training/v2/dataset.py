from pathlib import Path
from typing import List, Optional

from .types import MediaAsset

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def discover_video_assets(input_path: Path, limit: Optional[int] = None) -> List[MediaAsset]:
    resolved = input_path.expanduser().resolve()
    paths: List[Path] = []

    if resolved.is_file() and resolved.suffix.lower() in VIDEO_EXTENSIONS:
        paths = [resolved]
    elif resolved.is_dir():
        paths = sorted(
            path
            for path in resolved.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and not path.name.startswith(".")
        )

    if limit is not None and limit > 0:
        paths = paths[:limit]

    assets: List[MediaAsset] = []
    for path in paths:
        source_id = f"video_{path.stem}"
        assets.append(MediaAsset(path=path, source_id=source_id))
    return assets
