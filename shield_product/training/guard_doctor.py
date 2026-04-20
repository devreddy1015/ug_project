import argparse
import importlib.util
import json
import os
import platform
from pathlib import Path
import shutil
import sys
from typing import Dict, List

from env_loader import load_env_files


MODULE_GROUPS: Dict[str, List[str]] = {
    "core": ["numpy", "PIL"],
    "ui": ["streamlit", "pandas"],
    "enhanced": ["cv2", "whisper", "pytesseract", "sentence_transformers", "chromadb", "torch"],
}

BINARY_GROUPS: Dict[str, List[str]] = {
    "media": ["ffmpeg", "ffprobe"],
    "ocr": ["tesseract"],
}

LLM_ENV_KEYS = ["GROQ_API_KEY", "GROQ_MODEL", "GROQ_API_BASE", "LLM_API_KEY", "LLM_MODEL", "LLM_API_BASE"]

MODULE_PACKAGE_HINTS: Dict[str, str] = {
    "PIL": "pillow",
    "chromadb": "chromadb",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "pandas": "pandas",
    "pytesseract": "pytesseract",
    "sentence_transformers": "sentence-transformers",
    "streamlit": "streamlit",
    "torch": "torch",
    "whisper": "openai-whisper",
}

BINARY_INSTALL_HINTS: Dict[str, str] = {
    "ffmpeg": "Install ffmpeg and ensure it is on PATH.",
    "ffprobe": "Install ffprobe (usually packaged with ffmpeg) and ensure it is on PATH.",
    "tesseract": "Install tesseract OCR binary and ensure it is on PATH.",
}

_LOADED_ENV_FILES = tuple(load_env_files(Path(__file__).resolve().parents[1]))


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _binary_available(binary_name: str) -> bool:
    return shutil.which(binary_name) is not None


def _collect_report() -> Dict[str, object]:
    modules: Dict[str, Dict[str, bool]] = {}
    for group, names in MODULE_GROUPS.items():
        modules[group] = {name: _module_available(name) for name in names}

    binaries: Dict[str, Dict[str, bool]] = {}
    for group, names in BINARY_GROUPS.items():
        binaries[group] = {name: _binary_available(name) for name in names}

    env = {key: bool(os.environ.get(key, "").strip()) for key in LLM_ENV_KEYS}

    core_ready = all(modules["core"].values())
    ui_ready = all(modules["ui"].values())
    media_ready = all(binaries["media"].values())
    llm_ready = env.get("GROQ_API_KEY", False) or env.get("LLM_API_KEY", False)

    missing_critical: List[str] = []
    if not core_ready:
        missing_critical.extend([name for name, ok in modules["core"].items() if not ok])

    missing_recommended: List[str] = []
    for group in ["ui", "enhanced"]:
        missing_recommended.extend([name for name, ok in modules[group].items() if not ok])
    for group in ["media", "ocr"]:
        missing_recommended.extend([name for name, ok in binaries[group].items() if not ok])
    if not llm_ready:
        missing_recommended.append("llm_api_key")

    module_names = {name for names in MODULE_GROUPS.values() for name in names}
    missing_module_packages = sorted(
        {
            MODULE_PACKAGE_HINTS.get(name, name)
            for name in missing_recommended
            if name in module_names
        }
    )
    missing_binary_names = sorted(
        {
            name
            for group in binaries.values()
            for name, is_available in group.items()
            if not is_available
        }
    )

    recommended_actions: List[str] = []
    if missing_module_packages:
        recommended_actions.append(
            f"{sys.executable} -m pip install {' '.join(missing_module_packages)}"
        )
    if not llm_ready:
        recommended_actions.append(
            "Set GROQ_API_KEY (or LLM_API_KEY) in .env or .env.service."
        )
    for binary_name in missing_binary_names:
        recommended_actions.append(
            BINARY_INSTALL_HINTS.get(binary_name, f"Install '{binary_name}' and ensure it is on PATH.")
        )

    return {
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "env_files_loaded": list(_LOADED_ENV_FILES),
        "modules": modules,
        "binaries": binaries,
        "llm_env": env,
        "readiness": {
            "core_ready": core_ready,
            "ui_ready": ui_ready,
            "media_ready": media_ready,
            "llm_ready": llm_ready,
        },
        "missing_critical": sorted(missing_critical),
        "missing_recommended": sorted(set(missing_recommended)),
        "recommended_actions": recommended_actions,
    }


def _print_text(report: Dict[str, object]) -> None:
    runtime = report["runtime"]
    readiness = report["readiness"]

    print("Guard Doctor Report")
    print(f"Python: {runtime['python_version']}")
    print(f"Executable: {runtime['python_executable']}")
    print(f"Platform: {runtime['platform']}")
    print()

    print("Loaded env files")
    if report["env_files_loaded"]:
        for path in report["env_files_loaded"]:
            print(f"- {path}")
    else:
        print("- none")
    print()

    print("Readiness")
    for key, value in readiness.items():
        print(f"- {key}: {'ok' if value else 'missing'}")

    print()
    print("Modules")
    for group, values in report["modules"].items():
        statuses = ", ".join(f"{name}={'ok' if ok else 'missing'}" for name, ok in values.items())
        print(f"- {group}: {statuses}")

    print()
    print("Binaries")
    for group, values in report["binaries"].items():
        statuses = ", ".join(f"{name}={'ok' if ok else 'missing'}" for name, ok in values.items())
        print(f"- {group}: {statuses}")

    print()
    print("LLM Env")
    env_statuses = ", ".join(
        f"{name}={'set' if is_set else 'unset'}" for name, is_set in report["llm_env"].items()
    )
    print(f"- {env_statuses}")

    if report["missing_critical"]:
        print()
        print("Critical missing:")
        for name in report["missing_critical"]:
            print(f"- {name}")

    if report["missing_recommended"]:
        print()
        print("Recommended missing:")
        for name in report["missing_recommended"]:
            print(f"- {name}")

    if report["recommended_actions"]:
        print()
        print("Recommended actions:")
        for action in report["recommended_actions"]:
            print(f"- {action}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard environment diagnostics")
    parser.add_argument("--json", action="store_true", help="Print report in JSON format")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when critical dependencies are missing",
    )
    parser.add_argument(
        "--strict-recommended",
        action="store_true",
        help="Exit non-zero when recommended dependencies are missing",
    )
    args = parser.parse_args()

    report = _collect_report()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)

    if args.strict and report["missing_critical"]:
        raise SystemExit(2)

    if args.strict_recommended and report["missing_recommended"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
