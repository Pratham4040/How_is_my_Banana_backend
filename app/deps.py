from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.domain.types import ImageProcessorProtocol, PredictorProtocol
from app.services.image_processor import ImageProcessor
from app.services.model_loader import get_model
from app.services.predictor import BananaPredictor


@lru_cache(maxsize=1)
def get_predictor() -> PredictorProtocol:
    settings = get_settings()
    model = get_model()
    processor = ImageProcessor(settings.image_target_size)
    return BananaPredictor(model=model, image_processor=processor)


@lru_cache(maxsize=1)
def get_image_processor() -> ImageProcessorProtocol:
    settings = get_settings()
    return ImageProcessor(settings.image_target_size)
