import logging
import logging.config
from typing import Any, Iterable

from quiz_generator.settings import settings


def init_logging(
    *,
    log_level: str | None = None,
    dep_log_level: str | None = None,
    noisy_loggers: Iterable[str] | None = None,
) -> None:
    """Initialize logging for the project.

    Configuration lives in `quiz_generator.settings.settings` and is loaded via
    Pydantic settings (env prefix: `QUIZ_GENERATOR_`).
    """

    update: dict[str, Any] = {}
    if log_level is not None:
        update["log_level"] = log_level.upper()
    if dep_log_level is not None:
        update["dep_log_level"] = dep_log_level.upper()
    if noisy_loggers is not None:
        update["noisy_loggers"] = list(noisy_loggers)

    effective_settings = settings.model_copy(update=update) if update else settings
    logging.config.dictConfig(effective_settings.logging_config)


def truncate_long_strings(obj: Any, max_length: int | None = None) -> Any:
    """Recursively truncate long strings in nested data structures."""

    limit = max_length if max_length is not None else settings.string_max_length

    if isinstance(obj, str) and len(obj) > limit:
        return obj[:limit] + f"... [+{len(obj) - limit} chars]"
    if isinstance(obj, dict):
        return {k: truncate_long_strings(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [truncate_long_strings(item, limit) for item in obj]
    return obj
