#!/usr/bin/env python3
"""Evaluate predictions or a trained model with SHIELD-style metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
SHIELD_PRODUCT_DIR = ROOT_DIR / "shield_product"
if str(SHIELD_PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(SHIELD_PRODUCT_DIR))


try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    print("scikit-learn/scipy not found. Install with: pip install scikit-learn scipy")
    raise SystemExit(1)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_THRESHOLD_PCT = 50.0

SHIELD_LLM = {
    "accuracy": 93.32,
    "precision": 83.92,
    "recall": 83.91,
    "f1": 0.84,
    "fpr": 0.0417,
    "roc_auc": 0.8391,
    "mae": 3.07,
    "mse": 15.18,
    "r2": 0.9843,
    "pearson": 0.9982,
    "spearman": 0.9995,
}


def load_results(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")
    print(f"Loaded {len(rows)} predictions from {csv_path}")
    return rows


def load_ground_truth(gt_path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with gt_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = (row.get("video_path") or row.get("path") or row.get("file") or "").strip()
            label = parse_label(row.get("label"))
            if path and label is not None:
                add_ground_truth_keys(mapping, path, label)
    print(f"Loaded {len(mapping)} ground-truth lookup keys from {gt_path}")
    return mapping


def load_dataset_ground_truth(dataset_dir: Path) -> dict[str, int]:
    dataset_dir = dataset_dir.expanduser().resolve()
    mapping: dict[str, int] = {}
    videos = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    for path in videos:
        label = infer_dataset_label(path)
        if label is None:
            continue
        add_ground_truth_keys(mapping, str(path), label, dataset_dir=dataset_dir)

    if not mapping:
        raise SystemExit(
            f"No labels could be inferred from {dataset_dir}. Expected folders/names like "
            "'safe', 'harmful', 'non_hate', or 'hate'."
        )
    print(f"Loaded labels for {len(videos)} dataset videos from {dataset_dir}")
    return mapping


def add_ground_truth_keys(
    mapping: dict[str, int],
    path_text: str,
    label: int,
    dataset_dir: Path | None = None,
) -> None:
    for key in match_keys(path_text, dataset_dir=dataset_dir):
        if not key:
            continue
        old = mapping.get(key)
        if old is None or old == label:
            mapping[key] = label
        else:
            mapping.pop(key, None)


def match_keys(path_text: str, dataset_dir: Path | None = None) -> set[str]:
    keys = {path_text}
    path = Path(path_text)
    keys.add(path.name)

    try:
        resolved = path.expanduser().resolve()
        keys.add(str(resolved))
        if dataset_dir is not None:
            dataset_resolved = dataset_dir.expanduser().resolve()
            try:
                keys.add(str(resolved.relative_to(dataset_resolved)))
            except ValueError:
                pass
    except OSError:
        pass

    return {key for key in keys if key}


def parse_label(value: object) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    try:
        numeric = float(normalized)
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
    except ValueError:
        pass
    if normalized in {"1", "true", "toxic", "harmful", "hate", "explicit", "block", "review"}:
        return 1
    if normalized in {"0", "false", "non_toxic", "nontoxic", "safe", "non_hate", "clean"}:
        return 0
    return None


def infer_dataset_label(path: Path) -> int | None:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "safe" in parts or "non_hate" in name or "non-hate" in name:
        return 0
    if "harmful" in parts or "hate" in name:
        return 1
    return None


def build_results_from_model(
    model_path: Path,
    dataset_dir: Path,
    run_summary_path: Path | None,
    threshold_pct: float,
    val_only: bool,
    val_split: float,
    seed: int,
) -> list[dict[str, object]]:
    from training.dataset import collect_items
    from training.inference import InferenceConfig, VideoAnalyzer
    from training.train import _resolve_split_indices

    dataset_dir = dataset_dir.expanduser().resolve()
    run_summary = read_json(run_summary_path) if run_summary_path else {}
    feature_block = run_summary.get("feature_spec") if isinstance(run_summary, dict) else {}
    frame_count = int(feature_block.get("frame_count", 4)) if isinstance(feature_block, dict) else 4
    use_frames = bool(feature_block.get("use_frames", True)) if isinstance(feature_block, dict) else True

    items, _feature_spec, _categories = collect_items(
        dataset_dir=dataset_dir,
        modality="video",
        multi_label=False,
        labels_path=None,
        frame_count=frame_count,
        use_frames=use_frames,
    )
    if not items:
        raise SystemExit(f"No labeled videos found in {dataset_dir}")

    if val_only:
        _train_indices, val_indices = _resolve_split_indices(items, False, val_split, seed)
        selected_items = [items[index] for index in val_indices]
        split_name = "validation"
    else:
        selected_items = items
        split_name = "all"

    config = InferenceConfig(
        model_path=model_path.expanduser().resolve(),
        run_summary_path=run_summary_path.expanduser().resolve() if run_summary_path else None,
        threshold=threshold_pct,
        preview_frames=0,
        save_previews=False,
    )
    analyzer = VideoAnalyzer(config)

    rows: list[dict[str, object]] = []
    scratch_dir = ROOT_DIR / "evaluation_outputs" / "_previews_disabled"
    total = len(selected_items)
    print(f"Running model on {total} {split_name} videos at threshold {threshold_pct:.2f}%")

    for index, item in enumerate(selected_items, start=1):
        result = analyzer.analyze_video(item.path, scratch_dir)
        result["split"] = split_name
        result["true_label"] = int(item.label.item())
        rows.append(result)
        if index % 50 == 0 or index == total:
            print(f"  scored {index}/{total}")

    return rows


def read_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_threshold_pct(
    requested: float | None,
    model_path: Path | None,
    run_summary_path: Path | None,
) -> float:
    if requested is not None:
        return normalize_threshold_pct(requested)

    summary = read_json(run_summary_path)
    if "recommended_threshold" in summary:
        return normalize_threshold_pct(summary["recommended_threshold"])

    if model_path is not None and model_path.exists():
        try:
            import torch

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and "recommended_threshold" in checkpoint:
                return normalize_threshold_pct(checkpoint["recommended_threshold"])
        except (OSError, RuntimeError, TypeError, ValueError):
            pass

    return DEFAULT_THRESHOLD_PCT


def normalize_threshold_pct(value: object) -> float:
    parsed = safe_float(value, DEFAULT_THRESHOLD_PCT)
    if parsed <= 1.0:
        parsed *= 100.0
    return max(0.0, min(parsed, 100.0))


def write_predictions(rows: list[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_path",
        "split",
        "true_label",
        "classification",
        "harm_score_pct",
        "threshold",
        "view_recommendation",
        "age_preference",
        "warning_notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            notes = row.get("warning_notes", [])
            if isinstance(notes, list):
                notes_text = "; ".join(str(note) for note in notes)
            else:
                notes_text = str(notes)
            writer.writerow(
                {
                    "video_path": row.get("video_path"),
                    "split": row.get("split", ""),
                    "true_label": row.get("true_label", ""),
                    "classification": row.get("classification"),
                    "harm_score_pct": row.get("harm_score_pct"),
                    "threshold": row.get("threshold"),
                    "view_recommendation": row.get("view_recommendation"),
                    "age_preference": row.get("age_preference"),
                    "warning_notes": notes_text,
                }
            )
    print(f"Saved predictions to {csv_path}")


def align(
    results: list[dict[str, object]],
    gt: dict[str, int],
    threshold_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred_bin: list[int] = []
    y_scores: list[float] = []

    missing = 0
    for row in results:
        path_text = str(row.get("video_path", "")).strip()
        label = None
        if "true_label" in row and str(row.get("true_label", "")).strip() != "":
            label = parse_label(row.get("true_label"))
        if label is None:
            for key in match_keys(path_text):
                if key in gt:
                    label = gt[key]
                    break
        if label is None:
            missing += 1
            continue

        score_pct = safe_float(row.get("harm_score_pct"), 0.0)
        pred_bin = parse_prediction(row.get("classification"), score_pct, threshold_pct)

        y_true.append(label)
        y_pred_bin.append(pred_bin)
        y_scores.append(max(0.0, min(score_pct / 100.0, 1.0)))

    if missing:
        print(f"Skipped {missing} predictions with no matching ground-truth label")
    if not y_true:
        raise SystemExit("No overlapping samples between predictions and ground truth.")

    print(f"Evaluating on {len(y_true)} samples")
    return np.array(y_true), np.array(y_pred_bin), np.array(y_scores)


def parse_prediction(value: object, score_pct: float, threshold_pct: float) -> int:
    label = parse_label(value)
    if label is not None:
        return label
    return 1 if score_pct >= threshold_pct else 0


def compute_metrics(y_true: np.ndarray, y_pred_bin: np.ndarray, y_scores: np.ndarray) -> dict[str, object]:
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float("nan")

    scores_100 = y_scores * 100.0
    true_100 = y_true * 100.0

    mae = mean_absolute_error(true_100, scores_100)
    mse = mean_squared_error(true_100, scores_100)
    r2 = r2_score(true_100, scores_100)
    pearson = safe_corr(pearsonr, true_100, scores_100)
    spearman = safe_corr(spearmanr, true_100, scores_100)

    return {
        "samples": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "roc_auc": round(float(auc), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "r2": round(float(r2), 4),
        "pearson": round(float(pearson), 4),
        "spearman": round(float(spearman), 4),
    }


def safe_corr(func, y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_scores)) < 2:
        return float("nan")
    value = func(y_true, y_scores)[0]
    return float(value)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def print_report(metrics: dict[str, object], threshold_pct: float) -> None:
    line = "=" * 64
    print(line)
    print("YOUR MODEL - SHIELD-STYLE EVALUATION REPORT")
    print(line)
    print(f"Samples: {metrics['samples']}  positives: {metrics['positives']}  negatives: {metrics['negatives']}")
    print(f"Decision threshold: {threshold_pct:.2f}%")

    print("\nClassification metrics")
    print_metric("Accuracy", metrics["accuracy"], "%", SHIELD_LLM["accuracy"])
    print_metric("Precision", metrics["precision"], "%", SHIELD_LLM["precision"])
    print_metric("Recall", metrics["recall"], "%", SHIELD_LLM["recall"])
    print_metric("F1 Score", metrics["f1"], "", SHIELD_LLM["f1"])
    print_metric("FPR", metrics["fpr"], "", SHIELD_LLM["fpr"], lower_is_better=True)
    print_metric("ROC AUC", metrics["roc_auc"], "", SHIELD_LLM["roc_auc"])

    ci = metrics.get("bootstrap_ci_95")
    if isinstance(ci, dict):
        accuracy_ci = ci.get("accuracy")
        f1_ci = ci.get("f1")
        if isinstance(accuracy_ci, list) and isinstance(f1_ci, list):
            print(
                f"Bootstrap 95% CI: accuracy [{accuracy_ci[0]:.2f}, {accuracy_ci[1]:.2f}], "
                f"F1 [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]"
            )

    print("\nConfusion matrix")
    print(f"TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

    print("\nRegression-style score metrics")
    print_metric("MAE", metrics["mae"], "", SHIELD_LLM["mae"], lower_is_better=True)
    print_metric("MSE", metrics["mse"], "", SHIELD_LLM["mse"], lower_is_better=True)
    print_metric("R2", metrics["r2"], "", SHIELD_LLM["r2"])
    print_metric("Pearson r", metrics["pearson"], "", SHIELD_LLM["pearson"])
    print_metric("Spearman rho", metrics["spearman"], "", SHIELD_LLM["spearman"])
    print("\nNote: your dataset has binary labels only, so MAE/MSE/R2/Pearson/Spearman are computed")
    print("against a 0-or-100 severity proxy. SHIELD's paper used human severity scores from 0 to 100.")
    print(line)


def print_metric(
    name: str,
    value: object,
    suffix: str,
    shield_value: float,
    lower_is_better: bool = False,
) -> None:
    numeric = safe_float(value, float("nan"))
    delta = numeric - shield_value
    if lower_is_better:
        better = numeric < shield_value
    else:
        better = numeric > shield_value
    marker = "better" if better else "worse"
    if np.isnan(numeric):
        print(f"{name:12}: nan{suffix:1}   SHIELD LLM: {shield_value}")
        return
    print(f"{name:12}: {numeric:.4g}{suffix:1}   SHIELD LLM: {shield_value}   ({marker}, delta {delta:+.4g})")


def save_report(metrics: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    else:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                writer.writerow([key, value])
    print(f"Saved report to {out_path}")


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_bin: np.ndarray,
    y_scores: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples <= 0:
        return {}

    rng = np.random.default_rng(seed)
    size = len(y_true)
    metric_values: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    for _ in range(samples):
        indices = rng.integers(0, size, size=size)
        sample_true = y_true[indices]
        sample_pred = y_pred_bin[indices]
        sample_scores = y_scores[indices]
        sampled = compute_metrics(sample_true, sample_pred, sample_scores)
        for key in metric_values:
            value = safe_float(sampled.get(key), float("nan"))
            if not np.isnan(value):
                metric_values[key].append(value)

    intervals: dict[str, list[float]] = {}
    for key, values in metric_values.items():
        if not values:
            continue
        lower, upper = np.percentile(np.array(values, dtype=float), [2.5, 97.5])
        intervals[key] = [round(float(lower), 4), round(float(upper), 4)]
    return intervals


def existing_default_summary(model_path: Path | None, explicit_path: str | None) -> Path | None:
    if explicit_path:
        return Path(explicit_path)
    if model_path is None:
        return None
    candidate = model_path.expanduser().resolve().parent / "run_summary.json"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model/results against SHIELD-style metrics")
    parser.add_argument("--results", type=str, default=None, help="CSV predictions to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Trained model.pt to run before evaluating")
    parser.add_argument("--run-summary", type=str, default=None, help="run_summary.json for the trained model")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset folder with safe/harmful labels")
    parser.add_argument("--gt", type=str, default=None, help="Ground-truth CSV with video_path,label")
    parser.add_argument("--val-only", action="store_true", help="Evaluate only the recreated validation split")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split used during training")
    parser.add_argument("--seed", type=int, default=42, help="Seed used during training split")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold as 0-1 or 0-100")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap resamples for 95%% CI")
    parser.add_argument("--bootstrap-seed", type=int, default=123, help="Seed for bootstrap confidence intervals")
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="evaluation_outputs/model_predictions.csv",
        help="Where to save predictions when --model is used",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="evaluation_outputs/eval_report.csv",
        help="Where to save metrics report (.csv or .json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model) if args.model else None
    run_summary_path = existing_default_summary(model_path, args.run_summary)
    threshold_pct = resolve_threshold_pct(args.threshold, model_path, run_summary_path)

    if model_path:
        if args.dataset is None:
            raise SystemExit("--dataset is required when using --model")
        results = build_results_from_model(
            model_path=model_path,
            dataset_dir=Path(args.dataset),
            run_summary_path=run_summary_path,
            threshold_pct=threshold_pct,
            val_only=bool(args.val_only),
            val_split=float(args.val_split),
            seed=int(args.seed),
        )
        write_predictions(results, Path(args.predictions_out))
    else:
        results_path = Path(args.results or "analysis_outputs/analysis_results.csv")
        if not results_path.exists():
            raise SystemExit(
                f"Results file not found: {results_path}. Provide --results, or use "
                "--model ... --dataset ... to generate predictions first."
            )
        results = load_results(results_path)

    if args.gt:
        ground_truth = load_ground_truth(Path(args.gt))
    elif args.dataset:
        ground_truth = load_dataset_ground_truth(Path(args.dataset))
    else:
        raise SystemExit("Provide --gt or --dataset so ground truth labels are available.")

    y_true, y_pred_bin, y_scores = align(results, ground_truth, threshold_pct)
    metrics = compute_metrics(y_true, y_pred_bin, y_scores)
    metrics["bootstrap_ci_95"] = bootstrap_ci(
        y_true,
        y_pred_bin,
        y_scores,
        samples=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
    )
    metrics["threshold_pct"] = round(threshold_pct, 4)
    metrics["shield_llm_reference"] = SHIELD_LLM
    print_report(metrics, threshold_pct)
    save_report(metrics, Path(args.report_out))


if __name__ == "__main__":
    main()
