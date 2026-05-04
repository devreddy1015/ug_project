from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _load_run_module():
    run_path = Path(__file__).resolve().parents[2] / "run.py"
    spec = importlib.util.spec_from_file_location("run", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load run.py from {run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_module():
    return _load_run_module()


def test_auto_detect_modality_by_extension(run_module) -> None:
    assert run_module.detect_modality(Path("input.txt"), "auto") == "text"
    assert run_module.detect_modality(Path("input.jpg"), "auto") == "image"
    assert run_module.detect_modality(Path("input.wav"), "auto") == "audio"
    assert run_module.detect_modality(Path("input.mp4"), "auto") == "video"


def test_auto_detect_unsupported_extension_raises(run_module) -> None:
    with pytest.raises(ValueError):
        run_module.detect_modality(Path("input.bin"), "auto")


@pytest.mark.parametrize(
    ("modality", "report"),
    [
        (
            "text",
            {
                "verdict": "BLOCK",
                "analysis_confidence": 91.0,
                "modality_scores": {"text": 88.0, "image": 95.0},
            },
        ),
        (
            "image",
            {
                "verdict": "REVIEW",
                "analysis_confidence": 73.0,
                "modality_scores": {"image": 76.0},
            },
        ),
        (
            "audio",
            {
                "verdict": "SAFE",
                "analysis_confidence": 84.0,
                "modality_scores": {"audio": 81.0, "text": 62.0},
            },
        ),
        (
            "video",
            {
                "verdict": "BLOCK",
                "analysis_confidence": 97.0,
                "modality_scores": {"video": 98.0, "image": 90.0, "text": 86.0},
            },
        ),
    ],
)
def test_run_inference_modalities_with_mocked_analyzers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_module,
    modality: str,
    report: dict,
) -> None:
    input_file = tmp_path / f"sample.{_extension_for_modality(modality)}"
    input_file.write_text("mock content for moderation", encoding="utf-8")

    if modality == "text":
        monkeypatch.setattr(run_module, "analyze_text_payload", lambda _: report)
    elif modality in {"image", "audio"}:
        monkeypatch.setattr(run_module, "analyze_non_video_file", lambda _path, modality: report)
    else:
        monkeypatch.setattr(run_module, "analyze_video_file", lambda _path: report)

    payload = run_module.run_inference(input_file, modality)

    assert payload["label"] == report["verdict"]
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["modality_scores"], dict)
    assert payload["modality_scores"]


def test_cli_integration_text_auto_mode(tmp_path: Path) -> None:
    input_file = tmp_path / "integration_input.txt"
    input_file.write_text("violent hate threat with dangerous cues", encoding="utf-8")

    run_path = Path(__file__).resolve().parents[2] / "run.py"
    env = os.environ.copy()
    env["GUARD_DISABLE_EMBEDDINGS"] = "1"
    env["GUARD_DISABLE_HATE_CLASSIFIER"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            str(run_path),
            "--input",
            str(input_file),
            "--modality",
            "auto",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)

    assert payload["label"] in {"SAFE", "REVIEW", "BLOCK"}
    assert isinstance(payload["confidence"], float)
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["modality_scores"], dict)


def _extension_for_modality(modality: str) -> str:
    return {
        "text": "txt",
        "image": "jpg",
        "audio": "wav",
        "video": "mp4",
    }[modality]
