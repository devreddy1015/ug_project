from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Tuple

from env_loader import load_env_files


_LOADED_ENV_FILES = tuple(load_env_files(Path(__file__).resolve().parents[1]))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, minimum)


def _parse_extensions(raw: str) -> Tuple[str, ...]:
    values = []
    for item in raw.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        values.append(normalized)
    return tuple(sorted(set(values)))


def _parse_csv(raw: str) -> Tuple[str, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(values)


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    api_prefix: str
    data_dir: Path
    database_path: Path
    redis_url: str
    log_level: str
    request_id_header: str
    require_api_key: bool
    api_key: str
    api_key_header: str
    max_upload_size_bytes: int
    default_page_size: int
    max_page_size: int
    task_soft_time_limit_sec: int
    task_time_limit_sec: int
    result_ttl_seconds: int
    allowed_origins: Tuple[str, ...]
    allowed_image_extensions: Tuple[str, ...]
    allowed_video_extensions: Tuple[str, ...]
    allowed_audio_extensions: Tuple[str, ...]
    webhook_timeout_seconds: int
    webhook_max_attempts: int
    webhook_user_agent: str
    guard_frame_count: int
    guard_enable_whisper: bool
    guard_enable_ocr: bool
    guard_region: str
    text_max_chars: int

    @property
    def allowed_extensions_by_modality(self) -> Dict[str, Tuple[str, ...]]:
        return {
            "image": self.allowed_image_extensions,
            "video": self.allowed_video_extensions,
            "audio": self.allowed_audio_extensions,
        }


def get_settings() -> Settings:
    app_name = os.environ.get("APP_NAME", "SHIELD Guard Service").strip() or "SHIELD Guard Service"
    environment = os.environ.get("APP_ENV", "development").strip().lower() or "development"
    api_prefix = os.environ.get("API_PREFIX", "/v1").strip() or "/v1"
    if not api_prefix.startswith("/"):
        api_prefix = f"/{api_prefix}"

    data_dir = Path(os.environ.get("DATA_DIR", "service_data")).expanduser().resolve()
    database_path = Path(
        os.environ.get("DATABASE_PATH", str(data_dir / "jobs.db"))
    ).expanduser().resolve()
    # Use a non-zero Redis DB by default to reduce queue collisions with other local apps.
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/15").strip()

    log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    request_id_header = os.environ.get("REQUEST_ID_HEADER", "X-Request-ID").strip() or "X-Request-ID"

    require_api_key = _env_bool("REQUIRE_API_KEY", False)
    api_key = os.environ.get("API_KEY", "").strip()
    api_key_header = os.environ.get("API_KEY_HEADER", "X-API-Key").strip() or "X-API-Key"

    if require_api_key and not api_key:
        raise ValueError("REQUIRE_API_KEY is enabled but API_KEY is missing")

    max_upload_size_bytes = _env_int("MAX_UPLOAD_SIZE_BYTES", 500 * 1024 * 1024)
    default_page_size = _env_int("DEFAULT_PAGE_SIZE", 20)
    max_page_size = _env_int("MAX_PAGE_SIZE", 100)
    task_soft_time_limit_sec = _env_int("TASK_SOFT_TIME_LIMIT_SEC", 900)
    task_time_limit_sec = _env_int("TASK_TIME_LIMIT_SEC", 1200)
    result_ttl_seconds = _env_int("RESULT_TTL_SECONDS", 7 * 24 * 60 * 60)
    allowed_origins = _parse_csv(os.environ.get("ALLOWED_ORIGINS", "*"))

    allowed_image_extensions = _parse_extensions(
        os.environ.get("ALLOWED_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp,.bmp")
    )
    allowed_video_extensions = _parse_extensions(
        os.environ.get("ALLOWED_VIDEO_EXTENSIONS", ".mp4,.mov,.avi,.mkv,.webm,.m4v")
    )
    allowed_audio_extensions = _parse_extensions(
        os.environ.get("ALLOWED_AUDIO_EXTENSIONS", ".mp3,.wav,.m4a,.aac,.ogg,.flac")
    )

    webhook_timeout_seconds = _env_int("WEBHOOK_TIMEOUT_SECONDS", 8)
    webhook_max_attempts = _env_int("WEBHOOK_MAX_ATTEMPTS", 2)
    webhook_user_agent = os.environ.get("WEBHOOK_USER_AGENT", "shield-guard-service/1.0").strip()

    guard_frame_count = _env_int("GUARD_FRAME_COUNT", 8)
    guard_enable_whisper = _env_bool("GUARD_ENABLE_WHISPER", True)
    guard_enable_ocr = _env_bool("GUARD_ENABLE_OCR", True)
    guard_region = os.environ.get("GUARD_REGION", "auto").strip() or "auto"
    text_max_chars = _env_int("TEXT_MAX_CHARS", 50000)

    return Settings(
        app_name=app_name,
        environment=environment,
        api_prefix=api_prefix,
        data_dir=data_dir,
        database_path=database_path,
        redis_url=redis_url,
        log_level=log_level,
        request_id_header=request_id_header,
        require_api_key=require_api_key,
        api_key=api_key,
        api_key_header=api_key_header,
        max_upload_size_bytes=max_upload_size_bytes,
        default_page_size=default_page_size,
        max_page_size=max_page_size,
        task_soft_time_limit_sec=task_soft_time_limit_sec,
        task_time_limit_sec=task_time_limit_sec,
        result_ttl_seconds=result_ttl_seconds,
        allowed_origins=allowed_origins,
        allowed_image_extensions=allowed_image_extensions,
        allowed_video_extensions=allowed_video_extensions,
        allowed_audio_extensions=allowed_audio_extensions,
        webhook_timeout_seconds=webhook_timeout_seconds,
        webhook_max_attempts=webhook_max_attempts,
        webhook_user_agent=webhook_user_agent,
        guard_frame_count=guard_frame_count,
        guard_enable_whisper=guard_enable_whisper,
        guard_enable_ocr=guard_enable_ocr,
        guard_region=guard_region,
        text_max_chars=text_max_chars,
    )


SETTINGS = get_settings()
