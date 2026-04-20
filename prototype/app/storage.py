from pathlib import Path
import uuid
from dataclasses import dataclass

from fastapi import UploadFile

from .config import SETTINGS


INPUTS_DIR_NAME = "inputs"
COPY_CHUNK_SIZE = 1024 * 1024


class UploadValidationError(ValueError):
    pass


class UploadTooLargeError(UploadValidationError):
    pass


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    original_filename: str
    content_type: str
    size_bytes: int


def ensure_storage() -> Path:
    inputs_dir = SETTINGS.data_dir / INPUTS_DIR_NAME
    inputs_dir.mkdir(parents=True, exist_ok=True)
    return inputs_dir


def _normalize_extension(filename: str) -> str:
    return Path(filename).suffix.lower().strip()


def _validate_upload(modality: str, upload: UploadFile) -> str:
    filename = Path(upload.filename or "upload.bin").name
    extension = _normalize_extension(filename)
    allowed = SETTINGS.allowed_extensions_by_modality.get(modality, tuple())
    if allowed and extension not in allowed:
        allowed_text = ", ".join(allowed)
        raise UploadValidationError(
            f"Unsupported file extension '{extension or '<none>'}' for modality '{modality}'. "
            f"Allowed: {allowed_text}"
        )
    return filename


def save_upload(upload: UploadFile, modality: str) -> StoredUpload:
    inputs_dir = ensure_storage()
    filename = _validate_upload(modality, upload)
    suffix = _normalize_extension(filename)
    dest = inputs_dir / f"{uuid.uuid4().hex}{suffix}"

    written = 0
    try:
        with dest.open("wb") as file_handle:
            while True:
                chunk = upload.file.read(COPY_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > SETTINGS.max_upload_size_bytes:
                    raise UploadTooLargeError(
                        f"Upload exceeds max size of {SETTINGS.max_upload_size_bytes} bytes"
                    )
                file_handle.write(chunk)
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise
    finally:
        try:
            upload.file.close()
        except Exception:
            pass

    return StoredUpload(
        path=dest,
        original_filename=filename,
        content_type=str(upload.content_type or "application/octet-stream"),
        size_bytes=written,
    )
