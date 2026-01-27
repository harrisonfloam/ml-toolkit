from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Project settings (intentionally minimal).

    This file was adapted from another project; most non-logging settings were
    intentionally removed to keep quiz-generator focused.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="QUIZ_GENERATOR_",
        extra="ignore",
    )

    # LLM / Ollama settings used by the local CLI agent.
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model_id: str = "mistral-small3.2:latest"

    # LLM / OpenAI settings used when --provider=openai.
    openai_chat_model_id: str = "gpt-4o-mini"

    # Logging settings
    log_level: LogLevel = "INFO"
    dep_log_level: LogLevel = "WARNING"
    string_max_length: int = 300
    noisy_loggers: list[str] = [
        "pdfminer",
        "openai",
        "openai._base_client",
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
    ]

    @property
    def logging_config(self) -> dict:
        """logging.config.dictConfig() spec."""

        loggers: dict[str, dict] = {
            "quiz_generator": {
                "level": self.log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "agent_framework": {
                "level": self.log_level,
                "handlers": ["console"],
                "propagate": False,
            },
        }

        for name in self.noisy_loggers:
            loggers[name] = {
                "level": self.dep_log_level,
                "handlers": ["console"],
                "propagate": False,
            }

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": self.log_level,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": self.log_level,
            },
            "loggers": loggers,
        }


settings = Settings()
