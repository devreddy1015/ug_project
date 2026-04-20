from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _iter_candidate_files(project_root: Path) -> Iterable[Path]:
    explicit = os.environ.get("ENV_FILE", "").strip()
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = project_root / explicit_path
        yield explicit_path.resolve()

    for name in (".env", ".env.service", ".env.local"):
        yield (project_root / name).resolve()


def _strip_inline_comment(value: str) -> str:
    if value.startswith(("'", '"')):
        return value
    comment_index = value.find(" #")
    if comment_index >= 0:
        return value[:comment_index].rstrip()
    return value


def _parse_env_file(file_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            continue

        value = _strip_inline_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        values[key] = value

    return values


def load_env_files(project_root: Path | None = None, override_existing: bool = False) -> List[str]:
    root = (project_root or Path(__file__).resolve().parent).resolve()
    loaded_files: List[str] = []
    seen: set[Path] = set()

    for env_file in _iter_candidate_files(root):
        if env_file in seen:
            continue
        seen.add(env_file)

        if not env_file.exists() or not env_file.is_file():
            continue

        try:
            values = _parse_env_file(env_file)
        except OSError:
            continue

        for key, value in values.items():
            if override_existing or key not in os.environ:
                os.environ[key] = value

        loaded_files.append(str(env_file))

    return loaded_files
