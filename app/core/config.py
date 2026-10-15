"""Application settings.

Strict, typed configuration loaded from environment variables and an optional
``.env`` file (never committed). Secrets are modelled as ``SecretStr`` so they
are redacted from logs and ``model_dump`` output.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Runtime ───────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    service_name: str = "v-call"

    # ── Database ──────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://vcall:vcall@localhost:5432/vcall"
    database_url_sync: str | None = None
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_echo: bool = False

    # ── Redis / broker ────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Secrets (required — app fails fast if unset) ──────────────────
    jwt_secret: SecretStr = Field(..., min_length=32)
    fernet_key: SecretStr = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # ── LLM providers ─────────────────────────────────────────────────
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: SecretStr | None = None
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None

    # ── Transcription ─────────────────────────────────────────────────
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    pyannote_token: SecretStr | None = None
    audio_chunk_seconds: int = 600
    audio_max_upload_mb: int = 512
    audio_artifacts_dir: str = "artifacts"

    # ── Security ──────────────────────────────────────────────────────
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    rate_limit_default: str = "60/minute"
    rate_limit_heavy: str = "10/minute"

    # ── Compliance (DPDP Act 2023 / GDPR) ─────────────────────────────
    raw_audio_retention_hours: int = 1
    scrub_transcript_pii: bool = True

    # ── Validation ────────────────────────────────────────────────────
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, value: object) -> object:
        """Accept a JSON string (env var) or an already-parsed list."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return value

    @field_validator("audio_chunk_seconds")
    @classmethod
    def _validate_chunk(cls, value: int) -> int:
        if value < 30:
            raise ValueError("audio_chunk_seconds must be >= 30")
        return value

    @model_validator(mode="after")
    def _check_production_safety(self) -> Settings:
        """Refuse placeholder secrets in production."""
        if self.environment == "production":
            for name in ("jwt_secret", "fernet_key"):
                secret = getattr(self, name).get_secret_value()
                if "replace" in secret.lower():
                    raise ValueError(
                        f"{name} is still a placeholder; set a real secret in production"
                    )
        return self

    # ── Derived flags ─────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        return self.environment == "development" and bool(self.environment)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton.

    ``jwt_secret``/``fernet_key`` are deliberately required (fail fast) and
    supplied by the environment — hence the ignore below.
    """
    return Settings()  # type: ignore


def clear_settings_cache() -> None:
    """Drop the cached settings (intended for tests)."""
    get_settings.cache_clear()
