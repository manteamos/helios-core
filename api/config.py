"""
Runtime configuration via environment variables.

All settings can be overridden by setting HELIOS_* environment variables.
Example: HELIOS_REDIS_URL=redis://redis:6379/0

Defaults are appropriate for local development with Redis on localhost.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HELIOS_", env_file=".env")

    redis_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_result_ttl_seconds: int = 86_400  # 24 hours
    google_maps_key: str = ""


settings = Settings()
