import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageStat

from Engine.utils import probe_media, tokenize


@dataclass(frozen=True)
class FeatureSpec:
    input_dim: int
    frame_count: int
    use_frames: bool
    modality: str
    category_terms: Dict[str, List[str]]


def load_category_terms() -> Dict[str, List[str]]:
    root = Path(__file__).resolve().parents[1]
    bag_path = root / "Engine" / "VectorHandler" / "CategoryBags" / "category_terms.json"
    if not bag_path.exists():
        return {}
    payload = json.loads(bag_path.read_text(encoding="utf-8"))
    return {key: list(value) for key, value in payload.items()}


def build_feature_spec(modality: str, frame_count: int, use_frames: bool) -> FeatureSpec:
    category_terms = load_category_terms()
    if modality == "text":
        input_dim = len(category_terms) + 1
    elif modality == "image":
        input_dim = 6
    elif modality == "audio":
        input_dim = 4
    else:
        input_dim = 6 + (frame_count * 3 if use_frames else 0)
    return FeatureSpec(
        input_dim=input_dim,
        frame_count=frame_count,
        use_frames=use_frames,
        modality=modality,
        category_terms=category_terms,
    )


def extract_features(path: Path, spec: FeatureSpec) -> List[float]:
    if spec.modality == "text":
        return _text_features(path, spec)
    if spec.modality == "image":
        return _image_features(path)
    if spec.modality == "audio":
        return _audio_features(path)
    return _video_features(path, spec)


def _text_features(path: Path, spec: FeatureSpec) -> List[float]:
    text = ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""
    tokens = tokenize(text)
    token_set = set(tokens)
    features: List[float] = []
    for category, terms in spec.category_terms.items():
        if not terms:
            features.append(0.0)
            continue
        matches = len(token_set.intersection(term.lower() for term in terms))
        features.append(matches / max(len(terms), 1))
    features.append(min(len(text) / 2000.0, 1.0))
    return features


def _image_features(path: Path) -> List[float]:
    width = height = 0.0
    mean_r = mean_g = mean_b = 0.0
    size_bytes = float(path.stat().st_size) if path.exists() else 0.0
    try:
        with Image.open(path) as image:
            width, height = image.size
            stat = ImageStat.Stat(image.convert("RGB"))
            mean_r, mean_g, mean_b = stat.mean
    except Exception:
        width = height = 0.0
    return [
        width / 1920.0,
        height / 1080.0,
        size_bytes / 5_000_000.0,
        mean_r / 255.0,
        mean_g / 255.0,
        mean_b / 255.0,
    ]


def _audio_features(path: Path) -> List[float]:
    meta = probe_media(path)
    duration = _safe_float(meta.get("duration"))
    size_bytes = float(path.stat().st_size) if path.exists() else 0.0
    has_audio = 1.0 if meta.get("has_audio") else 0.0
    return [
        min(duration / 600.0, 1.0),
        min(size_bytes / 5_000_000.0, 1.0),
        has_audio,
        0.0,
    ]


def _video_features(path: Path, spec: FeatureSpec) -> List[float]:
    meta = probe_media(path)
    duration = _safe_float(meta.get("duration"))
    width = _safe_float(meta.get("width"))
    height = _safe_float(meta.get("height"))
    size_bytes = float(path.stat().st_size) if path.exists() else 0.0
    has_audio = 1.0 if meta.get("has_audio") else 0.0
    has_video = 1.0 if meta.get("has_video") else 0.0

    features = [
        min(duration / 600.0, 1.0),
        width / 1920.0,
        height / 1080.0,
        min(size_bytes / 50_000_000.0, 1.0),
        has_audio,
        has_video,
    ]

    if spec.use_frames:
        features.extend(_frame_mean_features(path, duration, spec.frame_count))

    return features


def _frame_mean_features(path: Path, duration: float, frame_count: int) -> List[float]:
    if shutil.which("ffmpeg") is None or frame_count <= 0 or duration <= 0:
        return [0.0] * (frame_count * 3)

    times = [duration * (idx + 1) / (frame_count + 1) for idx in range(frame_count)]
    means: List[float] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, timestamp in enumerate(times):
            frame_path = Path(temp_dir) / f"frame_{idx}.jpg"
            cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.2f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ]
            try:
                completed = subprocess.run(
                    cmd, capture_output=True, text=True, check=False, timeout=20
                )
                if completed.returncode != 0 or not frame_path.exists():
                    means.extend([0.0, 0.0, 0.0])
                    continue
                with Image.open(frame_path) as image:
                    stat = ImageStat.Stat(image.convert("RGB"))
                    mean_r, mean_g, mean_b = stat.mean
                    means.extend([mean_r / 255.0, mean_g / 255.0, mean_b / 255.0])
            except (subprocess.TimeoutExpired, Exception):
                means.extend([0.0, 0.0, 0.0])

    return means


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
