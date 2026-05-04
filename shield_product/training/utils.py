from __future__ import annotations


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(float(lower), min(float(value), float(upper)))
