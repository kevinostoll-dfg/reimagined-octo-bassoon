import json
import logging
import os
from typing import Any, Dict


def configure_logging(verbose: bool = True, log_level_env: str = "LOG_LEVEL") -> None:
    level_name = os.getenv(log_level_env)
    if level_name:
        level = logging._nameToLevel.get(level_name.upper(), logging.INFO)
    else:
        level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)
    logging.getLogger("llama_index.core.node_parser.node_utils").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)
    logging.getLogger("grpc._cython.cygrpc").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def redact_secret(value: str, show_last: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= show_last:
        return "*" * len(value)
    return "*" * (len(value) - show_last) + value[-show_last:]


def log_payload(logger: logging.Logger, message: str, payload: Dict[str, Any], level: int = logging.DEBUG) -> None:
    try:
        serialized = json.dumps(payload, default=str, ensure_ascii=True)
    except TypeError:
        serialized = str(payload)
    logger.log(level, "%s | payload=%s", message, serialized)
