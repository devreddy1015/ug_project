import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from Engine.utils import probe_media

from .features import build_feature_spec, extract_features
from .labels import infer_binary_label

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _collect_videos(dataset_dir: Path) -> List[Path]:
    videos: List[Path] = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return sorted(videos)


def _write_csv(path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return

    max_features = max(len(record.get("feature_vector", [])) for record in records)
    fieldnames = [
        "video_path",
        "label",
        "duration",
        "width",
        "height",
        "has_audio",
        "has_video",
        "size_bytes",
    ] + [f"feature_{idx}" for idx in range(max_features)]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            metadata = record.get("metadata", {})
            features = record.get("feature_vector", [])
            row = {
                "video_path": record.get("video_path"),
                "label": record.get("label"),
                "duration": metadata.get("duration") if isinstance(metadata, dict) else None,
                "width": metadata.get("width") if isinstance(metadata, dict) else None,
                "height": metadata.get("height") if isinstance(metadata, dict) else None,
                "has_audio": metadata.get("has_audio") if isinstance(metadata, dict) else None,
                "has_video": metadata.get("has_video") if isinstance(metadata, dict) else None,
                "size_bytes": record.get("size_bytes"),
            }
            for idx, value in enumerate(features):
                row[f"feature_{idx}"] = value
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess videos into model-ready features")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="preprocess_videos.json")
    parser.add_argument("--output-csv", type=str, default="preprocess_videos.csv")
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    feature_spec = build_feature_spec(
        modality="video", frame_count=args.frame_count, use_frames=not args.no_frames
    )

    videos = _collect_videos(dataset_dir)
    if args.limit is not None:
        videos = videos[: args.limit]

    records: List[Dict[str, object]] = []
    for path in videos:
        feature_vector = extract_features(path, feature_spec)
        metadata = probe_media(path)
        records.append(
            {
                "video_path": str(path),
                "label": infer_binary_label(path),
                "feature_vector": feature_vector,
                "feature_dim": len(feature_vector),
                "metadata": metadata,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": str(dataset_dir),
        "count": len(records),
        "feature_spec": {
            "input_dim": feature_spec.input_dim,
            "frame_count": feature_spec.frame_count,
            "use_frames": feature_spec.use_frames,
        },
        "items": records,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_csv, records)

    print(f"Preprocessing complete: {len(records)} videos")
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()
