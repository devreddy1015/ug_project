"""Compatibility shim for legacy imports."""

from .guard_analyze import *  # noqa: F401,F403
from .guard_analyze import main


if __name__ == "__main__":
    main()
