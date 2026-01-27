import logging
import logging.config
from typing import Any


def init_logging(log_level: str = "INFO", logging_config: dict | None = None) -> None:
    """
    Initialize logging using the configuration defined in settings.logging_config.
    """
    log_level = log_level.upper()
    if logging_config is not None:
        config = logging_config.copy()
    else:
        logging.basicConfig(level=log_level)
        return
    config["root"]["level"] = log_level
    for handler in config["handlers"].values():
        handler["level"] = log_level
    logging.config.dictConfig(config)


def truncate_long_strings(obj: Any, max_length: int = 1000) -> Any:
    """Recursively truncate long strings in nested data structures."""
    if isinstance(obj, str) and len(obj) > max_length:
        return obj[:max_length] + f"... [+{len(obj) - max_length} chars]"
    elif isinstance(obj, dict):
        return {k: truncate_long_strings(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_long_strings(item, max_length) for item in obj]
    return obj
