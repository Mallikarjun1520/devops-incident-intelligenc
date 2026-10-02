"""Shared logger configuration."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
