import logging


def configure_logging(level: str) -> None:
    normalized = (level or "INFO").upper()
    root = logging.getLogger()

    if not root.handlers:
        logging.basicConfig(
            level=normalized,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        root.setLevel(normalized)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "celery"):
        logging.getLogger(name).setLevel(normalized)
