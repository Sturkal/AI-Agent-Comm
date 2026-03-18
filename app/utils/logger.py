"""Central logging configuration."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name or "sap_sfim_ai_agent")
    if not logger.handlers:
        handler = logging.StreamHandler()
        if os.getenv("LOG_FORMAT", "text").strip().lower() == "json":
            formatter: logging.Formatter = JsonFormatter()
        else:
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").strip().upper())
        logger.propagate = False
    return logger


def configure_root_logger() -> logging.Logger:
    """Ensure the root logger follows the same format/level for production containers."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return root_logger

    handler = logging.StreamHandler()
    if os.getenv("LOG_FORMAT", "text").strip().lower() == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").strip().upper())
    return root_logger
