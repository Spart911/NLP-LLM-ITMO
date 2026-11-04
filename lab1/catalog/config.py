"""Чтение ключа, адреса провайдера и имени модели из окружения."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_REQUIRED = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


def _env_file() -> Path | None:
    start = Path.cwd()
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    packaged = Path(__file__).resolve().parents[2] / ".env"
    return packaged if packaged.is_file() else None


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str

    def __repr__(self) -> str:
        return (
            f"Settings(base_url={self.base_url!r}, model={self.model!r}, api_key='***')"
        )


def get_settings() -> Settings:
    env_file = _env_file()
    if env_file is not None:
        load_dotenv(env_file)
    missing = [name for name in _REQUIRED if not os.getenv(name, "").strip()]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Не заданы переменные окружения: {names}")
    return Settings(
        api_key=os.environ["LLM_API_KEY"].strip(),
        base_url=os.environ["LLM_BASE_URL"].strip().rstrip("/"),
        model=os.environ["LLM_MODEL"].strip(),
    )


def redact(text: str) -> str:
    secret = os.getenv("LLM_API_KEY", "").strip()
    if secret:
        text = text.replace(secret, "***")
    return text
