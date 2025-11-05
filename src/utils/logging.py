import logging
import logging.config
from typing import Any

from src.generative.settings import settings


def init_logging(log_level: str = settings.log_level):
    """
    Initialize logging using the configuration defined in settings.logging_config.
    """
    log_level = log_level.upper()
    config = settings.logging_config.copy()
    config["root"]["level"] = log_level
    for handler in config["handlers"].values():
        handler["level"] = log_level
    logging.config.dictConfig(config)


def truncate_long_strings(
    obj: Any, max_length: int = settings.string_max_length
) -> Any:
    """Recursively truncate long strings in nested data structures."""
    if isinstance(obj, str) and len(obj) > max_length:
        return obj[:max_length] + f"... [+{len(obj) - max_length} chars]"
    elif isinstance(obj, dict):
        return {k: truncate_long_strings(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_long_strings(item, max_length) for item in obj]
    return obj
