"""
core/config.py — Application Configuration
============================================
Centralised settings management using Pydantic-Settings.

All configuration values are read from environment variables (or a .env file),
making the application 12-factor compliant and deployment-environment agnostic.

Usage
-----
    from core.config import settings

    api_key = settings.GROQ_API_KEY
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralised, type-safe application settings.

    Values are resolved in the following priority order:
    1. Environment variables (highest priority)
    2. Variables defined in the `.env` file
    3. Default values defined here (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently discard unknown env vars
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_ENV: str = Field(
        default="development",
        description="Runtime environment: development | staging | production",
    )
    APP_NAME: str = Field(
        default="AI for the Indian Investor",
        description="Human-readable application name.",
    )
    APP_VERSION: str = Field(default="0.1.0", description="Semantic version.")
    DEBUG: bool = Field(default=False, description="Enable debug mode.")

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins (comma-separated in env var).",
    )

    # ── Groq / LLM ────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key used by LangChain-Groq agents.",
    )
    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default Groq chat-completion model.",
    )
    GROQ_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM responses (0 = deterministic).",
    )

    # ── NSE / Market Data ─────────────────────────────────────────────────────
    NSE_BASE_URL: str = Field(
        default="https://www.nseindia.com",
        description="Base URL for NSE data endpoints.",
    )
    NSE_REQUEST_TIMEOUT: int = Field(
        default=10,
        ge=1,
        description="HTTP timeout (seconds) for NSE API calls.",
    )

    # ── Database (placeholder for future use) ─────────────────────────────────
    DATABASE_URL: str = Field(
        default="sqlite:///./nse_intelligence.db",
        description="SQLAlchemy-compatible database connection string.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("APP_ENV")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        allowed = {"development", "staging", "production"}
        if value.lower() not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}, got '{value}'")
        return value.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{value}'")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    The ``@lru_cache`` decorator ensures the .env file is parsed only once
    during the application lifetime, avoiding redundant I/O on every import.
    """
    return Settings()


# Module-level singleton for convenient import
settings: Settings = get_settings()
