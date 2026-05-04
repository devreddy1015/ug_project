import importlib
import json
import logging
import os
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
from training.utils import clamp, safe_float


logger = logging.getLogger(__name__)

DEFAULT_CLIP_PROMPTS: Dict[str, Tuple[str, ...]] = {
    "hate_speech": (
        "hateful symbol",
        "racist visual content",
        "violent hate message",
        "harassing hateful meme",
    ),
    "explicit_content": (
        "explicit adult nudity",
        "pornographic content",
        "sexual explicit scene",
    ),
    "violence": (
        "graphic violence",
        "weapon attack scene",
        "bloody injury",
    ),
    "self_harm": (
        "self harm act",
        "suicidal behavior",
        "dangerous self injury",
    ),
}


_OPTIONAL_IMPORT_CACHE: Dict[str, object | None] = {}
_WHISPER_MODEL_CACHE: Dict[str, object] = {}
_CLIP_BUNDLE_CACHE: Dict[str, Tuple[object, object, object, str]] = {}


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
    keyframe_interval_seconds: float = 2.0,
    whisper_model: str = "tiny",
    clip_model: str = "openai/clip-vit-base-patch32",
    enable_whisper: bool = True,
    enable_ocr: bool = True,
) -> ReelSignalBundle:
    resolved = video_path.expanduser().resolve()
    metadata = dict(probe_media(resolved))
    diagnostics = _extractor_diagnostics(enable_whisper=enable_whisper, enable_ocr=enable_ocr)
    resolved_whisper_model = (
        str(whisper_model).strip()
        or os.environ.get("GUARD_WHISPER_MODEL_NAME", "tiny").strip()
        or "tiny"
    )
    resolved_clip_model = (
        str(clip_model).strip()
        or os.environ.get("GUARD_CLIP_MODEL_NAME", "openai/clip-vit-base-patch32").strip()
        or "openai/clip-vit-base-patch32"
    )

    sampled_frames, frame_signals = _sample_frames_with_opencv(
        resolved,
        frame_count,
        keyframe_interval_seconds=keyframe_interval_seconds,
    )
    frame_backend = "opencv" if sampled_frames else "none"
    if not sampled_frames:
        sampled_frames, frame_signals = _sample_frames_with_ffmpeg(
            resolved,
            frame_count,
            keyframe_interval_seconds=keyframe_interval_seconds,
        )
        if sampled_frames:
            frame_backend = "ffmpeg"

    transcript_text = (
        transcribe_audio_file(resolved, model_name=resolved_whisper_model)
        if enable_whisper
        else ""
    )
    ocr_text = _extract_ocr_text(sampled_frames) if enable_ocr else ""
    caption_text, hashtags = _read_caption_and_hashtags(resolved)
    sidecar = _read_sidecar_context(resolved)
    clip_scores = _max_pool_clip_scores(sampled_frames, clip_model=resolved_clip_model)

    if clip_scores:
        frame_signals.update({f"clip_{name}_score": score for name, score in clip_scores.items()})
        frame_signals["clip_max_risk"] = round(max(clip_scores.values()), 4)
    else:
        frame_signals["clip_max_risk"] = 0.0

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

    clip_context = _build_clip_context(clip_scores)
    if clip_context:
        combined_text = "\n".join(part for part in [combined_text, clip_context] if part).strip()

    duration = safe_float(metadata.get("duration"))
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


def _sample_frames_with_opencv(
    video_path: Path,
    frame_count: int,
    keyframe_interval_seconds: float,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
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

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = float(frame_total / fps) if fps > 0 else 0.0

    indices: List[int] = []
    if duration > 0 and keyframe_interval_seconds > 0:
        cursor = max(float(keyframe_interval_seconds), 0.5)
        max_samples = max(int(frame_count), 1) * 12
        while cursor < duration and len(indices) < max_samples:
            frame_index = int(cursor * fps)
            if 0 <= frame_index < frame_total:
                indices.append(frame_index)
            cursor += max(float(keyframe_interval_seconds), 0.5)

    if not indices:
        indices = [
            int((idx + 1) * frame_total / (max(frame_count, 1) + 1))
            for idx in range(max(frame_count, 1))
        ]

    if frame_count > 0 and len(indices) > frame_count:
        step = len(indices) / float(frame_count)
        indices = [indices[int(idx * step)] for idx in range(frame_count)]

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


def _sample_frames_with_ffmpeg(
    video_path: Path,
    frame_count: int,
    keyframe_interval_seconds: float,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    if shutil.which("ffmpeg") is None:
        return [], _empty_frame_signals()

    meta = probe_media(video_path)
    duration = safe_float(meta.get("duration"))
    if duration <= 0:
        return [], _empty_frame_signals()

    timestamps: List[float] = []
    if keyframe_interval_seconds > 0:
        cursor = max(float(keyframe_interval_seconds), 0.5)
        max_samples = max(int(frame_count), 1) * 12
        while cursor < duration and len(timestamps) < max_samples:
            timestamps.append(cursor)
            cursor += max(float(keyframe_interval_seconds), 0.5)

    if not timestamps:
        effective_count = max(int(frame_count), 1)
        timestamps = [duration * (idx + 1) / (effective_count + 1) for idx in range(effective_count)]

    if frame_count > 0 and len(timestamps) > frame_count:
        step = len(timestamps) / float(frame_count)
        timestamps = [timestamps[int(idx * step)] for idx in range(frame_count)]

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
                    if completed.returncode != 0 and completed.stderr:
                        logger.debug(
                            "ffmpeg keyframe extraction failed for %s at %.2fs: %s",
                            video_path,
                            timestamp,
                            completed.stderr.strip(),
                        )
                    continue
                with Image.open(frame_path) as image:
                    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
                    frames.append(rgb[:, :, ::-1])
            except (subprocess.TimeoutExpired, OSError, ValueError, RuntimeError) as exc:
                logger.debug(
                    "ffmpeg keyframe extraction raised %s for %s at %.2fs",
                    type(exc).__name__,
                    video_path,
                    timestamp,
                )
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
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("ocr extraction failed for one frame: %s", exc)
            continue
    return " ".join(texts)


def transcribe_audio_file(video_path: Path, model_name: str = "tiny") -> str:
    model = _get_whisper_model(model_name)
    if model is None:
        logger.info("whisper model unavailable for transcription (%s)", model_name)
        return ""

    try:
        result = model.transcribe(str(video_path))
        return str(result.get("text", "")).strip()
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.warning(
            "whisper transcription failed for %s with model %s: %s",
            video_path,
            model_name,
            exc,
        )
        return ""


def score_image_clip(
    image_path: Path,
    clip_model: str = "openai/clip-vit-base-patch32",
) -> Dict[str, float]:
    try:
        with Image.open(image_path) as image:
            return _score_clip_image(image.convert("RGB"), clip_model=clip_model)
    except OSError as exc:
        logger.warning("failed to open image for CLIP scoring at %s: %s", image_path, exc)
    except ValueError as exc:
        logger.warning("invalid image payload for CLIP scoring at %s: %s", image_path, exc)
    return {}


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
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            logger.debug("failed to read sidecar context %s: %s", candidate, exc)
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
        except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as exc:
            logger.debug("failed to parse sidecar caption file %s: %s", candidate, exc)
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


def _max_pool_clip_scores(frames: List[np.ndarray], clip_model: str) -> Dict[str, float]:
    if not frames:
        return {}

    pooled: Dict[str, float] = {}
    for frame in frames:
        try:
            image = Image.fromarray(frame[:, :, ::-1])
        except (TypeError, ValueError):
            continue

        frame_scores = _score_clip_image(image, clip_model=clip_model)
        for category, value in frame_scores.items():
            pooled[category] = max(pooled.get(category, 0.0), float(value))

    return {name: round(clamp(score), 4) for name, score in pooled.items()}


def _score_clip_image(image: Image.Image, clip_model: str) -> Dict[str, float]:
    clip_bundle = _get_clip_bundle(clip_model)
    if clip_bundle is None:
        return {}

    processor, model, torch, device = clip_bundle
    prompt_texts: List[str] = []
    prompt_labels: List[str] = []
    for category, prompts in DEFAULT_CLIP_PROMPTS.items():
        for prompt in prompts:
            prompt_texts.append(prompt)
            prompt_labels.append(category)

    if not prompt_texts:
        return {}

    try:
        inputs = processor(
            text=prompt_texts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.warning("CLIP inference failed: %s", exc)
        return {}

    category_scores: Dict[str, float] = {}
    for probability, category in zip(probabilities, prompt_labels):
        category_scores[category] = max(category_scores.get(category, 0.0), float(probability) * 100.0)

    return {name: round(clamp(score), 4) for name, score in category_scores.items()}


def _build_clip_context(clip_scores: Dict[str, float]) -> str:
    if not clip_scores:
        return ""

    risk_descriptors = []
    if float(clip_scores.get("hate_speech", 0.0)) >= 25.0:
        risk_descriptors.append("visual_hate_cues")
    if float(clip_scores.get("explicit_content", 0.0)) >= 25.0:
        risk_descriptors.append("visual_explicit_content")
    if float(clip_scores.get("violence", 0.0)) >= 25.0:
        risk_descriptors.append("visual_violence")
    if float(clip_scores.get("self_harm", 0.0)) >= 25.0:
        risk_descriptors.append("visual_self_harm")

    if not risk_descriptors:
        return ""

    formatted_scores = ", ".join(
        f"{name}={float(score):.1f}" for name, score in sorted(clip_scores.items())
    )
    return "visual_clip_signals: " + " ".join(risk_descriptors) + f" ({formatted_scores})"


def _get_clip_bundle(clip_model: str) -> Optional[Tuple[object, object, object, str]]:
    if _env_flag("GUARD_DISABLE_CLIP"):
        return None

    cached = _CLIP_BUNDLE_CACHE.get(clip_model)
    if cached is not None:
        return cached

    transformers = _optional_import("transformers")
    torch = _optional_import("torch")
    if transformers is None or torch is None:
        return None

    device = _resolve_torch_device(torch)

    try:
        processor = transformers.CLIPProcessor.from_pretrained(clip_model)
        model = transformers.CLIPModel.from_pretrained(clip_model)
        model.to(device)
        model.eval()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("failed to initialize CLIP model %s: %s", clip_model, exc)
        return None

    bundle = (processor, model, torch, device)
    _CLIP_BUNDLE_CACHE[clip_model] = bundle
    return bundle


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
    if _env_flag("GUARD_DISABLE_WHISPER"):
        return None

    torch = _optional_import("torch")
    device = _resolve_torch_device(torch)
    cache_key = f"{model_name}@{device}"

    cached = _WHISPER_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    whisper = _optional_import("whisper")
    if whisper is None:
        return None

    try:
        try:
            model = whisper.load_model(model_name, device=device)
        except TypeError:
            model = whisper.load_model(model_name)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("failed to initialize whisper model %s: %s", model_name, exc)
        return None

    _WHISPER_MODEL_CACHE[cache_key] = model
    return model


def _resolve_torch_device(torch_module: object | None) -> str:
    if torch_module is not None and bool(getattr(torch_module, "cuda", None)):
        if torch_module.cuda.is_available():
            return "cuda"
    return "cpu"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
    duration = safe_float(metadata.get("duration"))
    brightness = safe_float(frame_signals.get("avg_brightness"))
    motion = safe_float(frame_signals.get("avg_motion"))

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
