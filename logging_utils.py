"""logging_utils.py — Shared logging setup."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class _UTCFormatter(logging.Formatter):
    def formatTime(
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = record.levelname.ljust(8)
        return super().format(record)


_FMT = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_UTCFormatter(_FMT))
logging.basicConfig(level=logging.INFO, handlers=[_handler])


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
