"""
Application configuration — loads from .env and exposes typed settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def _list_int(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # ── LLM ──────────────────────────────────────────────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./meeting_mind.db")

    # ── Trello ───────────────────────────────────────────────────────────
    trello_api_key: str = os.getenv("TRELLO_API_KEY", "")
    trello_token: str = os.getenv("TRELLO_TOKEN", "")
    trello_board_id: str = os.getenv("TRELLO_BOARD_ID", "")
    trello_list_id: str = os.getenv("TRELLO_LIST_ID", "")

    # ── SMTP ─────────────────────────────────────────────────────────────
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "")

    # ── App ──────────────────────────────────────────────────────────────
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    app_secret_key: str = os.getenv("APP_SECRET_KEY", "dev-secret-change-me")
    reminder_hours_before: list[int] = field(
        default_factory=lambda: _list_int(os.getenv("REMINDER_HOURS_BEFORE", "24,2"))
    )
    digest_day: str = os.getenv("DIGEST_DAY", "monday")
    digest_hour: int = int(os.getenv("DIGEST_HOUR", "9"))

    # ── Derived ──────────────────────────────────────────────────────────
    @property
    def trello_configured(self) -> bool:
        return bool(self.trello_api_key and self.trello_token and self.trello_board_id)

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
