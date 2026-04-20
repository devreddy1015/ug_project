import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from Engine.utils import probe_media

from .features import FeatureSpec, build_feature_spec, extract_features
from .models import MLPClassifier

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    run_summary_path: Optional[Path]
    threshold: float
    preview_frames: int
    save_previews: bool


class VideoAnalyzer:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = _safe_torch_load(config.model_path)
        state_dict = checkpoint.get("model", checkpoint)
        input_dim = checkpoint.get("input_dim") or int(state_dict["net.0.weight"].shape[1])
        output_dim = _resolve_output_dim(checkpoint, state_dict)
        hidden_dims = _resolve_hidden_dims(checkpoint, state_dict)
        dropout = _resolve_dropout(checkpoint)

        run_summary = self._load_run_summary(config.run_summary_path)
        self.categories = self._resolve_categories(run_summary, output_dim)
        self.multi_label = output_dim > 1
        self.feature_spec = self._resolve_feature_spec(run_summary, input_dim)

        self.model = MLPClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def analyze_input(self, input_path: Path, output_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        videos = self._collect_videos(input_path)
        results: List[Dict[str, object]] = []

        for video_path in videos:
            results.append(self.analyze_video(video_path, output_dir))

        summary = self._build_summary(results)
        self._write_outputs(output_dir, results, summary)
        return results, summary

    def analyze_video(self, video_path: Path, output_dir: Path) -> Dict[str, object]:
        features = extract_features(video_path, self.feature_spec)
        fixed_features = self._resize_features(features, self.feature_spec.input_dim)

        tensor = torch.tensor([fixed_features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().flatten().tolist()

        metadata = probe_media(video_path)
        preview_paths: List[str] = []
        if self.config.save_previews:
            preview_paths = self._extract_preview_frames(video_path, output_dir, metadata)

        if self.multi_label:
            category_scores = {
                self.categories[idx]: round(prob * 100.0, 4)
                for idx, prob in enumerate(probs)
            }
            harm_score = max(category_scores.values()) if category_scores else 0.0
            top_categories = [
                f"{name} ({score:.1f})"
                for name, score in sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
                if score >= 30.0
            ]
        else:
            toxicity = round(probs[0] * 100.0, 4) if probs else 0.0
            category_scores = {"toxicity": toxicity}
            harm_score = toxicity
            top_categories = ["toxicity"] if toxicity >= 30.0 else []

        policy = self._policy_from_score(harm_score, top_categories)

        return {
            "video_path": str(video_path),
            "model_type": "multi_label" if self.multi_label else "binary",
            "threshold": self.config.threshold,
            "harm_score_pct": harm_score,
            "classification": "toxic" if harm_score >= self.config.threshold else "non_toxic",
            "category_scores_pct": category_scores,
            "metadata": metadata,
            "view_recommendation": policy["view_recommendation"],
            "age_preference": policy["age_preference"],
            "warning_notes": policy["warning_notes"],
            "preview_frames": preview_paths,
        }

    def _collect_videos(self, input_path: Path) -> List[Path]:
        resolved = input_path.expanduser().resolve()
        if resolved.is_file() and resolved.suffix.lower() in VIDEO_EXTENSIONS:
            return [resolved]
        if not resolved.exists():
            return []

        videos: List[Path] = []
        for path in resolved.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
        return sorted(videos)

    def _build_summary(self, results: List[Dict[str, object]]) -> Dict[str, object]:
        if not results:
            return {
                "total_videos": 0,
                "toxic_count": 0,
                "avg_harm_score_pct": 0.0,
                "max_harm_score_pct": 0.0,
                "max_harm_video": None,
                "recommendation_counts": {},
            }

        scores = [float(result["harm_score_pct"]) for result in results]
        toxic_count = sum(1 for result in results if result["classification"] == "toxic")
        recommendation_counts: Dict[str, int] = {}
        for result in results:
            key = str(result["view_recommendation"])
            recommendation_counts[key] = recommendation_counts.get(key, 0) + 1

        max_index = max(range(len(results)), key=lambda index: scores[index])
        return {
            "total_videos": len(results),
            "toxic_count": toxic_count,
            "avg_harm_score_pct": round(sum(scores) / len(scores), 4),
            "max_harm_score_pct": round(scores[max_index], 4),
            "max_harm_video": results[max_index]["video_path"],
            "recommendation_counts": recommendation_counts,
        }

    def _policy_from_score(self, harm_score_pct: float, warnings: List[str]) -> Dict[str, object]:
        if harm_score_pct >= 80.0:
            return {
                "view_recommendation": "do_not_show",
                "age_preference": "18_plus",
                "warning_notes": warnings[:6] or ["Severe toxicity risk detected"],
            }
        if harm_score_pct >= 60.0:
            return {
                "view_recommendation": "strong_caution",
                "age_preference": "18_plus",
                "warning_notes": warnings[:6] or ["High toxicity risk detected"],
            }
        if harm_score_pct >= 35.0:
            return {
                "view_recommendation": "caution",
                "age_preference": "16_plus",
                "warning_notes": warnings[:6] or ["Moderate toxicity risk detected"],
            }
        if harm_score_pct >= 20.0:
            return {
                "view_recommendation": "mild_caution",
                "age_preference": "13_plus",
                "warning_notes": warnings[:6] or ["Mild risk indicators present"],
            }
        return {
            "view_recommendation": "safe",
            "age_preference": "all_ages",
            "warning_notes": warnings[:6],
        }

    def _write_outputs(
        self,
        output_dir: Path,
        results: List[Dict[str, object]],
        summary: Dict[str, object],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis_results.json").write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
        (output_dir / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self._write_csv(output_dir / "analysis_results.csv", results)

    def _write_csv(self, csv_path: Path, results: List[Dict[str, object]]) -> None:
        fieldnames = [
            "video_path",
            "classification",
            "harm_score_pct",
            "view_recommendation",
            "age_preference",
            "warning_notes",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "video_path": result.get("video_path"),
                        "classification": result.get("classification"),
                        "harm_score_pct": result.get("harm_score_pct"),
                        "view_recommendation": result.get("view_recommendation"),
                        "age_preference": result.get("age_preference"),
                        "warning_notes": "; ".join(result.get("warning_notes", [])),
                    }
                )

    def _extract_preview_frames(
        self,
        video_path: Path,
        output_dir: Path,
        metadata: Dict[str, object],
    ) -> List[str]:
        if shutil.which("ffmpeg") is None or self.config.preview_frames <= 0:
            return []

        duration = _safe_float(metadata.get("duration"))
        if duration <= 0:
            return []

        preview_dir = output_dir / "previews" / video_path.stem
        preview_dir.mkdir(parents=True, exist_ok=True)

        timepoints = [
            duration * (idx + 1) / (self.config.preview_frames + 1)
            for idx in range(self.config.preview_frames)
        ]
        written: List[str] = []

        for idx, timestamp in enumerate(timepoints, start=1):
            frame_path = preview_dir / f"frame_{idx}.jpg"
            cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.2f}",
                "-i",
                str(video_path),
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
                if completed.returncode == 0 and frame_path.exists():
                    written.append(str(frame_path))
            except subprocess.TimeoutExpired:
                continue

        return written

    def _resolve_feature_spec(self, run_summary: Dict[str, object], input_dim: int) -> FeatureSpec:
        feature_block = run_summary.get("feature_spec") if isinstance(run_summary, dict) else None
        frame_count = 4
        use_frames = True

        if isinstance(feature_block, dict):
            frame_count = int(feature_block.get("frame_count", frame_count))
            use_frames = bool(feature_block.get("use_frames", use_frames))
        else:
            inferred_frame_count, inferred_use_frames = _infer_frames_from_input_dim(input_dim)
            frame_count = inferred_frame_count
            use_frames = inferred_use_frames

        return build_feature_spec("video", frame_count=frame_count, use_frames=use_frames)

    def _resolve_categories(self, run_summary: Dict[str, object], output_dim: int) -> List[str]:
        categories = []
        if isinstance(run_summary, dict):
            loaded = run_summary.get("categories")
            if isinstance(loaded, list):
                categories = [str(item) for item in loaded]

        if output_dim == 1:
            return ["toxicity"]

        if len(categories) == output_dim:
            return categories

        return [f"category_{index + 1}" for index in range(output_dim)]

    def _load_run_summary(self, path: Optional[Path]) -> Dict[str, object]:
        if path is None:
            return {}
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            return {}
        try:
            return json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _resize_features(self, features: List[float], target_dim: int) -> List[float]:
        if len(features) == target_dim:
            return features
        if len(features) > target_dim:
            return features[:target_dim]
        return features + [0.0] * (target_dim - len(features))


def _infer_frames_from_input_dim(input_dim: int) -> Tuple[int, bool]:
    if input_dim <= 6:
        return 0, False
    extra = input_dim - 6
    if extra % 3 == 0:
        frame_count = max(0, extra // 3)
        return frame_count, frame_count > 0
    return 4, True


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_torch_load(model_path: Path) -> Dict[str, object]:
    try:
        loaded = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(model_path, map_location="cpu")

    if isinstance(loaded, dict):
        return loaded
    return {"model": loaded}


def _resolve_output_dim(checkpoint: Dict[str, object], state_dict: Dict[str, object]) -> int:
    output_dim = checkpoint.get("output_dim")
    if output_dim is not None:
        try:
            parsed = int(output_dim)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass

    last_layer_index = -1
    last_output_dim = 1
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if not key.startswith("net.") or not key.endswith(".weight"):
            continue

        parts = key.split(".")
        if len(parts) != 3:
            continue

        try:
            layer_index = int(parts[1])
        except ValueError:
            continue

        if value.ndim == 2 and layer_index >= last_layer_index:
            last_layer_index = layer_index
            last_output_dim = int(value.shape[0])

    return max(last_output_dim, 1)


def _resolve_hidden_dims(checkpoint: Dict[str, object], state_dict: Dict[str, object]) -> Tuple[int, int]:
    encoded = checkpoint.get("hidden_dims")
    if isinstance(encoded, (list, tuple)) and len(encoded) >= 2:
        try:
            first = int(encoded[0])
            second = int(encoded[1])
            if first > 0 and second > 0:
                return first, second
        except (TypeError, ValueError):
            pass

    first_layer = state_dict.get("net.0.weight")
    second_layer = state_dict.get("net.3.weight")
    if isinstance(first_layer, torch.Tensor) and isinstance(second_layer, torch.Tensor):
        first = int(first_layer.shape[0])
        second = int(second_layer.shape[0])
        if first > 0 and second > 0:
            return first, second

    return 128, 64


def _resolve_dropout(checkpoint: Dict[str, object]) -> float:
    encoded = checkpoint.get("dropout")
    try:
        parsed = float(encoded)
    except (TypeError, ValueError):
        return 0.2
    return max(0.0, min(parsed, 0.8))
