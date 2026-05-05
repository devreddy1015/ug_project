import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


class LogMonitor:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.csv_path = self.output_dir / "storage.csv"

    def publish(self, payload: Dict[str, object]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        row = self._to_row(payload)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _to_row(self, payload: Dict[str, object]) -> Dict[str, object]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file_path": payload.get("file_path"),
            "modality": payload.get("modality"),
            "label": payload.get("label"),
            "harm_score": payload.get("harm_score"),
            "harm_score_pct": payload.get("harm_score_pct"),
            "view_recommendation": payload.get("view_recommendation"),
            "age_preference": payload.get("age_preference"),
            "warning_notes": json.dumps(payload.get("warning_notes", [])),
            "llm_scores": json.dumps(payload.get("llm_scores", {})),
            "llm_scores_pct": json.dumps(payload.get("llm_scores_pct", {})),
            "vector_scores": json.dumps(payload.get("vector_scores", {})),
            "vector_scores_pct": json.dumps(payload.get("vector_scores_pct", {})),
            "combined_scores": json.dumps(payload.get("combined_scores", {})),
            "combined_scores_pct": json.dumps(payload.get("combined_scores_pct", {})),
        }
