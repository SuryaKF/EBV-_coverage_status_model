from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    artifact_root_uri: str = os.getenv("ARTIFACT_ROOT_URI", "file://./artifacts")
    feature_version: str = os.getenv("FEATURE_VERSION", "v1")
    api_key: str = os.getenv("API_KEY", "")


settings = Settings()


def require_database_url() -> str:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be set")
    return settings.database_url
