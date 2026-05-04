from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parent
SHIELD_PRODUCT_DIR = ROOT_DIR / "shield_product"
if str(SHIELD_PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(SHIELD_PRODUCT_DIR))

from service.analyzers import analyze_non_video_file, analyze_text_payload, analyze_video_file
from service.config import SETTINGS


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".htm",
    ".yaml",
    ".yml",
    ".log",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHIELD moderation pipeline")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument(
        "--modality",
        choices=("auto", "text", "image", "audio", "video"),
        default="auto",
        help="Input modality or auto-detect by extension",
    )
    return parser.parse_args(argv)


def detect_modality(input_path: Path, requested_modality: str) -> str:
    requested = requested_modality.strip().lower()
    if requested and requested != "auto":
        return requested

    extension = input_path.suffix.lower()
    if extension in TEXT_EXTENSIONS:
        return "text"
    if extension in SETTINGS.allowed_image_extensions:
        return "image"
    if extension in SETTINGS.allowed_audio_extensions:
        return "audio"
    if extension in SETTINGS.allowed_video_extensions:
        return "video"

    raise ValueError(
        f"Could not auto-detect modality for extension '{extension or '<none>'}'. "
        "Pass --modality explicitly."
    )


def run_inference(input_path: Path, modality: str) -> Dict[str, Any]:
    if modality == "text":
        payload = input_path.read_text(encoding="utf-8", errors="ignore")
        if not payload.strip():
            raise ValueError(f"Input text file is empty: {input_path}")
        report = analyze_text_payload(payload)
    elif modality == "video":
        report = analyze_video_file(input_path)
    elif modality in {"image", "audio"}:
        report = analyze_non_video_file(input_path, modality=modality)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return _format_output(report)


def _format_output(report: Dict[str, Any]) -> Dict[str, Any]:
    label = _resolve_label(report)
    confidence = _resolve_confidence(report)
    modality_scores = _resolve_modality_scores(report)

    return {
        "label": label,
        "confidence": confidence,
        "modality_scores": modality_scores,
    }


def _resolve_label(report: Dict[str, Any]) -> str:
    verdict = str(report.get("verdict", "")).strip().upper()
    if verdict in {"SAFE", "REVIEW", "BLOCK"}:
        return verdict

    risk_score = _safe_float(report.get("overall_risk_score_out_of_100"))
    if risk_score >= 80.0:
        return "BLOCK"
    if risk_score >= 55.0:
        return "REVIEW"
    return "SAFE"


def _resolve_confidence(report: Dict[str, Any]) -> float:
    raw_confidence = _safe_float(report.get("analysis_confidence"))
    if raw_confidence <= 0.0:
        fallback = 100.0 - _safe_float(report.get("overall_risk_score_out_of_100"))
        raw_confidence = max(0.0, min(fallback, 100.0))

    if raw_confidence > 1.0:
        raw_confidence = raw_confidence / 100.0
    return round(max(0.0, min(raw_confidence, 1.0)), 4)


def _resolve_modality_scores(report: Dict[str, Any]) -> Dict[str, float]:
    raw_scores = report.get("modality_scores")
    normalized: Dict[str, float] = {}

    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            score = _safe_float(value)
            if score > 1.0:
                score = score / 100.0
            score = max(0.0, min(score, 1.0))
            if score > 0.0:
                normalized[str(key)] = round(score, 4)

    if normalized:
        return normalized

    fallback_modality = str(report.get("modality", "")).strip().lower()
    fallback_risk = _safe_float(report.get("overall_risk_score_out_of_100"))
    fallback_score = round(max(0.0, min(fallback_risk / 100.0, 1.0)), 4)
    if fallback_modality in {"text", "image", "audio", "video"}:
        return {fallback_modality: fallback_score}
    return {}


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        print(json.dumps({"error": f"Input file not found: {input_path}"}), file=sys.stderr)
        return 2

    try:
        modality = detect_modality(input_path, args.modality)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_inference(input_path, modality)
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
