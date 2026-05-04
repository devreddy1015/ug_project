import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def infer_binary_label(path: Path) -> Optional[int]:
    lowered = str(path).lower()
    if "non_hate" in lowered or "non-hate" in lowered:
        return 0
    if "hate" in lowered:
        return 1
    return None


def load_label_map(labels_path: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    if labels_path.suffix.lower() == ".csv":
        return _load_csv(labels_path)
    return _load_json(labels_path)


def find_labels_for_path(
    path: Path, dataset_dir: Path, label_map: Dict[str, Dict[str, float]]
) -> Optional[Dict[str, float]]:
    candidates = [
        str(path),
        path.name,
        str(path.relative_to(dataset_dir)) if path.is_relative_to(dataset_dir) else None,
    ]
    for candidate in candidates:
        if candidate and candidate in label_map:
            return label_map[candidate]
    return None


def _load_json(labels_path: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    mapping: Dict[str, Dict[str, float]] = {}

    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            file_key = entry.get("file") or entry.get("path") or entry.get("name")
            labels = entry.get("labels") or {}
            if file_key:
                mapping[file_key] = _normalize_labels(labels)
    elif isinstance(payload, dict):
        if "labels" in payload and isinstance(payload["labels"], dict):
            payload = payload["labels"]
        for file_key, labels in payload.items():
            mapping[str(file_key)] = _normalize_labels(labels)

    categories = _collect_categories(mapping.values())
    return mapping, categories


def _load_csv(labels_path: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    mapping: Dict[str, Dict[str, float]] = {}
    with labels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_key = row.get("file") or row.get("path") or row.get("name")
            label = row.get("label")
            if not file_key or not label:
                continue
            score = _parse_score(row.get("score"))
            mapping.setdefault(file_key, {})[label] = score

    categories = _collect_categories(mapping.values())
    return mapping, categories


def _collect_categories(label_sets: Iterable[Dict[str, float]]) -> List[str]:
    categories = set()
    for labels in label_sets:
        categories.update(labels.keys())
    return sorted(categories)


def _parse_score(value: Optional[str]) -> float:
    try:
        parsed = float(value) if value is not None else 0.0
    except ValueError:
        parsed = 0.0
    return _normalize_score(parsed)


def _normalize_labels(labels: object) -> Dict[str, float]:
    if not isinstance(labels, dict):
        return {}
    normalized: Dict[str, float] = {}
    for key, value in labels.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        normalized[str(key)] = _normalize_score(score)
    return normalized


def _normalize_score(score: float) -> float:
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(score, 1.0))
