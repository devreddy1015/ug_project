import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .FileSegregator import segregate_files
from .TaskPerformer import TaskPerformer
from Runner.Monitor.LogMonitor import LogMonitor


def _compute_metrics(results: List[Dict[str, object]], threshold: float) -> Dict[str, object]:
    tp = fp = tn = fn = 0
    for result in results:
        label = result.get("label")
        if label is None:
            continue
        pred = 1 if float(result.get("harm_score_pct", 0.0)) >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHIELD dataset runner")
    parser.add_argument("--dataset", type=str, default=os.environ.get("DATASET_DIR", ""))
    parser.add_argument("--output-dir", type=str, default="run_outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=35.0)
    parser.add_argument("--enable-transcription", action="store_true")
    parser.add_argument("--enable-ocr", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = segregate_files(dataset_dir, max_files=args.limit)
    monitor = LogMonitor(output_dir=output_dir)
    performer = TaskPerformer(
        monitor=monitor,
        enable_transcription=args.enable_transcription,
        enable_ocr=args.enable_ocr,
    )

    results: List[Dict[str, object]] = []
    start = datetime.now(timezone.utc)
    for item in files:
        results.append(performer.process_item(item))
    end = datetime.now(timezone.utc)

    metrics = _compute_metrics(results, args.threshold)
    run_summary = {
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "total_files": len(files),
        "threshold": args.threshold,
        "dataset_dir": str(dataset_dir),
    }

    hate_results = [entry for entry in results if entry.get("label") == 1]
    non_hate_results = [entry for entry in results if entry.get("label") == 0]

    _write_json(output_dir / "run_summary.json", run_summary)
    _write_json(output_dir / "quality_summary.json", metrics)
    _write_json(output_dir / "hate_results.json", {"results": hate_results})
    _write_json(output_dir / "non_hate_results.json", {"results": non_hate_results})


if __name__ == "__main__":
    main()
