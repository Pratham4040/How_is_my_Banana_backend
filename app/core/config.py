from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    host: str
    port: int
    log_level: str
    cors_origins: List[str]
    model_path: str
    image_target_size: int
    max_upload_mb: int


DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:10000",
    "https://howismybanana.netlify.app",
    "https://how-is-my-banana-backend.onrender.com",
]


def _parse_cors_origins(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_CORS_ORIGINS
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ORIGINS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Model")
    default_model_path = os.path.join(model_dir, "bananaV2.h5")

    return Settings(
        app_name=os.getenv("APP_NAME", "How is my BANANA"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        cors_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS")),
        model_path=os.getenv("MODEL_PATH", default_model_path),
        image_target_size=int(os.getenv("IMAGE_TARGET_SIZE", "256")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "5")),
    )
