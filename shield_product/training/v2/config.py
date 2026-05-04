from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


yaml = importlib.import_module("yaml") if importlib.util.find_spec("yaml") else None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_int(value: object, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default

def _as_float(value: object, default: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, minimum)


@dataclass(frozen=True)
class FusionWeights:
    text: float = 0.6
    image: float = 0.2
    audio: float = 0.1
    video: float = 0.1


@dataclass(frozen=True)
class GuardV2Config:
    frame_count: int = 8
    keyframe_interval_seconds: float = 2.0
    enable_whisper: bool = True
    enable_ocr: bool = True
    region: str = "auto"

    safe_threshold: float = 35.0
    review_threshold: float = 55.0
    block_threshold: float = 80.0
    network_block_threshold: float = 75.0

    top_categories: int = 6

    use_learned_fusion: bool = True
    fusion_model_path: str = "training/v2/fusion_head.joblib"
    fusion_blend_weight: float = 0.35
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)

    sentence_model_name: str = "all-MiniLM-L6-v2"
    hate_classifier_model_name: str = "cardiffnlp/twitter-roberta-base-hate"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    whisper_model_name: str = "tiny"


    @classmethod
    def from_sources(
        cls,
        *,
        config_path: Optional[Path] = None,
        overrides: Optional[Mapping[str, object]] = None,
    ) -> "GuardV2Config":
        resolved_path = (config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
        loaded = _load_yaml_config(resolved_path)
        guard = loaded.get("guard", {}) if isinstance(loaded, dict) else {}
        thresholds = guard.get("thresholds", {}) if isinstance(guard, dict) else {}
        fusion = guard.get("fusion", {}) if isinstance(guard, dict) else {}
        extraction = guard.get("extraction", {}) if isinstance(guard, dict) else {}
        models = guard.get("models", {}) if isinstance(guard, dict) else {}
        fallback_weights = fusion.get("fallback_weights", {}) if isinstance(fusion, dict) else {}

        fusion_weights = FusionWeights(
            text=_as_float(fallback_weights.get("text"), 0.40),
            image=_as_float(fallback_weights.get("image"), 0.22),
            audio=_as_float(fallback_weights.get("audio"), 0.18),
            video=_as_float(fallback_weights.get("video"), 0.20),
        )

        config = cls(
            frame_count=_as_int(extraction.get("frame_count"), 8),
            keyframe_interval_seconds=_as_float(extraction.get("keyframe_interval_seconds"), 2.0, minimum=0.5),
            enable_whisper=_as_bool(extraction.get("enable_whisper"), True),
            enable_ocr=_as_bool(extraction.get("enable_ocr"), True),
            region=str(guard.get("region", "auto")),
            safe_threshold=_as_float(thresholds.get("safe"), 35.0),
            review_threshold=_as_float(thresholds.get("review"), 55.0),
            block_threshold=_as_float(thresholds.get("block"), 80.0),
            network_block_threshold=_as_float(thresholds.get("network_block"), 75.0),
            top_categories=_as_int(guard.get("top_categories"), 6),
            use_learned_fusion=_as_bool(fusion.get("use_learned"), True),
            fusion_model_path=str(fusion.get("model_path", "training/v2/fusion_head.joblib")),
            fusion_blend_weight=_as_float(fusion.get("blend_weight"), 0.35),
            fusion_weights=fusion_weights,
            sentence_model_name=str(models.get("sentence_transformer", "all-MiniLM-L6-v2")),
            hate_classifier_model_name=str(
                models.get("hate_classifier", "cardiffnlp/twitter-roberta-base-hate")
            ),
            clip_model_name=str(models.get("clip_model", "openai/clip-vit-base-patch32")),
            whisper_model_name=str(models.get("whisper_model", "tiny")),
        )

        if not overrides:
            return config

        valid_fields = set(cls.__dataclass_fields__.keys())
        patch: Dict[str, object] = {}
        for key, value in overrides.items():
            if key in valid_fields and value is not None:
                patch[key] = value

        if not patch:
            return config
        return replace(config, **patch)


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    if yaml is None:
        logger.warning(
            "config.yaml present at %s but PyYAML is unavailable; using defaults",
            config_path,
        )
        return {}

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("failed to read config.yaml at %s: %s", config_path, exc)
        return {}

    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("failed to parse config.yaml at %s: %s", config_path, exc)
        return {}

    if isinstance(loaded, dict):
        return loaded
    return {}


def load_guard_v2_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> GuardV2Config:
    return GuardV2Config.from_sources(config_path=config_path, overrides=overrides)
