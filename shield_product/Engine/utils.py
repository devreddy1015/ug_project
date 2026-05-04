import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [token for token in cleaned.split() if token]


def safe_read_text(path: Path, max_chars: int = 20000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(max_chars)
    except (OSError, UnicodeError):
        return ""


def probe_media(path: Path) -> Dict[str, object]:
    result = {
        "duration": None,
        "width": None,
        "height": None,
        "has_audio": None,
        "has_video": None,
    }
    if shutil.which("ffprobe") is None:
        return result

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return result
        payload = json.loads(completed.stdout or "{}")
        format_info = payload.get("format", {})
        duration = format_info.get("duration")
        if duration is not None:
            try:
                result["duration"] = float(duration)
            except (TypeError, ValueError):
                result["duration"] = None
        streams = payload.get("streams", [])
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        has_video = any(stream.get("codec_type") == "video" for stream in streams)
        result["has_audio"] = has_audio
        result["has_video"] = has_video
        for stream in streams:
            if stream.get("codec_type") == "video":
                result["width"] = stream.get("width")
                result["height"] = stream.get("height")
                break
    except (OSError, json.JSONDecodeError):
        return result

    return result
