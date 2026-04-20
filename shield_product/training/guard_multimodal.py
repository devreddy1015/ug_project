import importlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from Engine.utils import probe_media


_OPTIONAL_IMPORT_CACHE: Dict[str, object | None] = {}
_WHISPER_MODEL_CACHE: Dict[str, object] = {}


@dataclass(frozen=True)
class ReelSignalBundle:
    video_path: str
    metadata: Dict[str, object]
    frame_signals: Dict[str, float]
    transcript_text: str
    ocr_text: str
    caption_text: str
    hashtags: List[str]
    combined_text: str
    engagement: Dict[str, float]
    creator_id: str
    region: str
    temporal_windows: List[Dict[str, object]]
    extraction_diagnostics: Dict[str, object]


def extract_reel_signals(
    video_path: Path,
    frame_count: int = 6,
    enable_whisper: bool = True,
    enable_ocr: bool = True,
) -> ReelSignalBundle:
    resolved = video_path.expanduser().resolve()
    metadata = dict(probe_media(resolved))
    diagnostics = _extractor_diagnostics(enable_whisper=enable_whisper, enable_ocr=enable_ocr)

    sampled_frames, frame_signals = _sample_frames_with_opencv(resolved, frame_count)
    frame_backend = "opencv" if sampled_frames else "none"
    if not sampled_frames:
        sampled_frames, frame_signals = _sample_frames_with_ffmpeg(resolved, frame_count)
        if sampled_frames:
            frame_backend = "ffmpeg"

    transcript_text = _transcribe_audio(resolved) if enable_whisper else ""
    ocr_text = _extract_ocr_text(sampled_frames) if enable_ocr else ""
    caption_text, hashtags = _read_caption_and_hashtags(resolved)
    sidecar = _read_sidecar_context(resolved)

    if not caption_text:
        caption_text = sidecar.get("caption_text", "")
    if not hashtags:
        sidecar_tags = sidecar.get("hashtags", [])
        if isinstance(sidecar_tags, list):
            hashtags = [str(tag).lstrip("#") for tag in sidecar_tags]

    combined_text = _join_text_sources(
        transcript_text=transcript_text,
        ocr_text=ocr_text,
        caption_text=caption_text,
        hashtags=hashtags,
    )

    duration = _safe_float(metadata.get("duration"))
    temporal_windows = _build_temporal_windows(transcript_text, duration, window_seconds=5)

    source_presence = {
        "has_transcript": bool(transcript_text.strip()),
        "has_ocr_text": bool(ocr_text.strip()),
        "has_caption_text": bool(caption_text.strip()),
        "has_hashtags": bool(hashtags),
        "has_combined_text": bool(combined_text.strip()),
    }
    diagnostics["frame_extraction_backend"] = frame_backend
    diagnostics["source_presence"] = source_presence
    diagnostics["text_lengths"] = {
        "transcript_chars": len(transcript_text),
        "ocr_chars": len(ocr_text),
        "caption_chars": len(caption_text),
        "combined_chars": len(combined_text),
    }
    diagnostics["insufficient_text_signal"] = not any(
        [
            source_presence["has_transcript"],
            source_presence["has_ocr_text"],
            source_presence["has_caption_text"],
            source_presence["has_hashtags"],
        ]
    )

    fallback_context = ""
    if diagnostics["insufficient_text_signal"]:
        fallback_context = _fallback_text_context(
            metadata=metadata,
            frame_signals=frame_signals,
        )
        if fallback_context:
            combined_text = "\n".join(part for part in [combined_text, fallback_context] if part).strip()

    source_presence["has_fallback_context"] = bool(fallback_context)
    diagnostics["inferred_profile"] = "unknown"
    diagnostics["fallback_context_applied"] = bool(fallback_context)
    diagnostics["text_lengths"]["combined_chars"] = len(combined_text)
    metadata["extraction_diagnostics"] = diagnostics

    engagement = _normalize_engagement(sidecar.get("engagement", {}))
    creator_id = str(sidecar.get("creator_id", resolved.parent.name or "unknown_creator"))
    region = str(sidecar.get("region", "global")).lower()

    return ReelSignalBundle(
        video_path=str(resolved),
        metadata=metadata,
        frame_signals=frame_signals,
        transcript_text=transcript_text,
        ocr_text=ocr_text,
        caption_text=caption_text,
        hashtags=hashtags,
        combined_text=combined_text,
        engagement=engagement,
        creator_id=creator_id,
        region=region,
        temporal_windows=temporal_windows,
        extraction_diagnostics=diagnostics,
    )


def _sample_frames_with_opencv(video_path: Path, frame_count: int) -> Tuple[List[np.ndarray], Dict[str, float]]:
    cv2 = _optional_import("cv2")
    if cv2 is None:
        return [], _empty_frame_signals()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], _empty_frame_signals()

    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_total <= 0:
        capture.release()
        return [], _empty_frame_signals()

    indices = [
        int((idx + 1) * frame_total / (frame_count + 1))
        for idx in range(max(frame_count, 1))
    ]

    frames: List[np.ndarray] = []
    brightness_values: List[float] = []
    motion_values: List[float] = []
    previous_gray: Optional[np.ndarray] = None

    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(gray.mean()))
        if previous_gray is not None:
            motion_values.append(float(np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32)))))
        previous_gray = gray
        frames.append(frame)

    capture.release()

    if not frames:
        return [], _empty_frame_signals()

    signals = {
        "sampled_frames": float(len(frames)),
        "avg_brightness": round(float(np.mean(brightness_values)), 4),
        "avg_motion": round(float(np.mean(motion_values)) if motion_values else 0.0, 4),
    }
    return frames, signals


def _sample_frames_with_ffmpeg(video_path: Path, frame_count: int) -> Tuple[List[np.ndarray], Dict[str, float]]:
    if shutil.which("ffmpeg") is None or frame_count <= 0:
        return [], _empty_frame_signals()

    meta = probe_media(video_path)
    duration = _safe_float(meta.get("duration"))
    if duration <= 0:
        return [], _empty_frame_signals()

    timestamps = [duration * (idx + 1) / (frame_count + 1) for idx in range(frame_count)]
    frames: List[np.ndarray] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, timestamp in enumerate(timestamps):
            frame_path = Path(temp_dir) / f"frame_{idx}.jpg"
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
                    cmd, capture_output=True, text=True, timeout=20, check=False
                )
                if completed.returncode != 0 or not frame_path.exists():
                    continue
                with Image.open(frame_path) as image:
                    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
                    frames.append(rgb[:, :, ::-1])
            except Exception:
                continue

    if not frames:
        return [], _empty_frame_signals()

    brightness_values = [float(np.mean(frame)) for frame in frames]
    signals = {
        "sampled_frames": float(len(frames)),
        "avg_brightness": round(float(np.mean(brightness_values)), 4),
        "avg_motion": 0.0,
    }
    return frames, signals


def _extract_ocr_text(frames: List[np.ndarray]) -> str:
    if not frames:
        return ""
    if shutil.which("tesseract") is None:
        return ""
    pytesseract = _optional_import("pytesseract")
    if pytesseract is None:
        return ""

    texts: List[str] = []
    for frame in frames[:6]:
        try:
            pil_frame = Image.fromarray(frame[:, :, ::-1])
            text = pytesseract.image_to_string(pil_frame)
            if text.strip():
                texts.append(text.strip())
        except Exception:
            continue
    return " ".join(texts)


def _transcribe_audio(video_path: Path) -> str:
    model_name = "base"
    model = _get_whisper_model(model_name)
    if model is None:
        return ""

    try:
        result = model.transcribe(str(video_path))
        return str(result.get("text", "")).strip()
    except Exception:
        return ""


def _read_sidecar_context(video_path: Path) -> Dict[str, object]:
    candidates = [
        video_path.with_suffix(".meta.json"),
        video_path.with_suffix(".context.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def _read_caption_and_hashtags(video_path: Path) -> Tuple[str, List[str]]:
    sidecar_candidates = [
        video_path.with_suffix(".caption.txt"),
        video_path.with_suffix(".txt"),
        video_path.with_suffix(".json"),
    ]

    caption = ""
    hashtags: List[str] = []

    for candidate in sidecar_candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".json":
                payload = json.loads(candidate.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(payload, dict):
                    caption = str(payload.get("caption", "") or payload.get("text", ""))
                    tag_values = payload.get("hashtags", [])
                    if isinstance(tag_values, list):
                        hashtags = [str(tag).lstrip("#") for tag in tag_values]
            else:
                caption = candidate.read_text(encoding="utf-8", errors="ignore")
            break
        except Exception:
            continue

    if not hashtags and caption:
        hashtags = re.findall(r"#([a-zA-Z0-9_]+)", caption)

    normalized_tags = sorted({tag.lower() for tag in hashtags if tag.strip()})
    return caption.strip(), normalized_tags


def _join_text_sources(
    transcript_text: str,
    ocr_text: str,
    caption_text: str,
    hashtags: List[str],
) -> str:
    parts = [
        transcript_text.strip(),
        ocr_text.strip(),
        caption_text.strip(),
        " ".join(f"#{tag}" for tag in hashtags),
    ]
    return "\n".join(part for part in parts if part)


def _normalize_engagement(payload: object) -> Dict[str, float]:
    if not isinstance(payload, dict):
        return {
            "likes": 0.0,
            "shares": 0.0,
            "comments": 0.0,
            "comment_sentiment": 0.0,
            "duets": 0.0,
            "stitches": 0.0,
        }

    def number(name: str) -> float:
        try:
            return max(0.0, float(payload.get(name, 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def signed_number(name: str) -> float:
        try:
            return float(payload.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0

    return {
        "likes": number("likes"),
        "shares": number("shares"),
        "comments": number("comments"),
        "comment_sentiment": max(-1.0, min(1.0, signed_number("comment_sentiment"))),
        "duets": number("duets"),
        "stitches": number("stitches"),
    }


def _build_temporal_windows(
    transcript_text: str, duration_seconds: float, window_seconds: int = 5
) -> List[Dict[str, object]]:
    if duration_seconds <= 0:
        return []

    window_count = max(1, int(np.ceil(duration_seconds / max(window_seconds, 1))))
    tokens = transcript_text.split()
    chunk_size = max(1, int(np.ceil(len(tokens) / window_count))) if tokens else 1

    windows: List[Dict[str, object]] = []
    for index in range(window_count):
        start = float(index * window_seconds)
        end = float(min(duration_seconds, (index + 1) * window_seconds))
        chunk_tokens = tokens[index * chunk_size : (index + 1) * chunk_size]
        chunk_text = " ".join(chunk_tokens).strip()
        recency_weight = round((index + 1) / window_count, 4)
        windows.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": chunk_text,
                "recency_weight": recency_weight,
            }
        )

    return windows


def _optional_import(module_name: str):
    if module_name in _OPTIONAL_IMPORT_CACHE:
        return _OPTIONAL_IMPORT_CACHE[module_name]

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        module = None

    _OPTIONAL_IMPORT_CACHE[module_name] = module
    return module


def _get_whisper_model(model_name: str):
    cached = _WHISPER_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    whisper = _optional_import("whisper")
    if whisper is None:
        return None

    try:
        model = whisper.load_model(model_name)
    except Exception:
        return None

    _WHISPER_MODEL_CACHE[model_name] = model
    return model


def _extractor_diagnostics(enable_whisper: bool, enable_ocr: bool) -> Dict[str, object]:
    has_cv2 = _optional_import("cv2") is not None
    has_ffmpeg = shutil.which("ffmpeg") is not None
    has_ffprobe = shutil.which("ffprobe") is not None
    has_whisper_module = _optional_import("whisper") is not None
    has_pytesseract_module = _optional_import("pytesseract") is not None
    has_tesseract_binary = shutil.which("tesseract") is not None

    missing_components: List[str] = []
    if not has_cv2 and not has_ffmpeg:
        missing_components.append("opencv_or_ffmpeg")
    if not has_ffprobe:
        missing_components.append("ffprobe")
    if enable_whisper and not has_whisper_module:
        missing_components.append("whisper")
    if enable_ocr and not has_pytesseract_module:
        missing_components.append("pytesseract")
    if enable_ocr and not has_tesseract_binary:
        missing_components.append("tesseract")

    return {
        "requested": {
            "whisper": bool(enable_whisper),
            "ocr": bool(enable_ocr),
        },
        "available": {
            "opencv": has_cv2,
            "ffmpeg": has_ffmpeg,
            "ffprobe": has_ffprobe,
            "whisper": has_whisper_module,
            "pytesseract": has_pytesseract_module,
            "tesseract": has_tesseract_binary,
        },
        "missing_components": missing_components,
    }


def _fallback_text_context(
    metadata: Dict[str, object],
    frame_signals: Dict[str, float],
) -> str:
    duration = _safe_float(metadata.get("duration"))
    brightness = _safe_float(frame_signals.get("avg_brightness"))
    motion = _safe_float(frame_signals.get("avg_motion"))

    media_hint = (
        f"duration_seconds={duration:.2f}; brightness={brightness:.2f}; motion={motion:.2f}."
    )
    return (
        "fallback_context profile=unknown: minimal text evidence available; "
        "request transcript/captions/OCR sidecars for reliable moderation and use conservative thresholds. "
        + media_hint
    )


def _empty_frame_signals() -> Dict[str, float]:
    return {"sampled_frames": 0.0, "avg_brightness": 0.0, "avg_motion": 0.0}


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
