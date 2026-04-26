from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.infrastructure.tf_model import TensorFlowModel


@lru_cache(maxsize=1)
def get_model() -> TensorFlowModel:
    settings = get_settings()
    return TensorFlowModel(settings.model_path)
