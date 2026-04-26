from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import HealthResponse, PredictionResponse
from app.core.exceptions import ImageDecodeError, InvalidImageError, PredictionError
from app.domain.types import ImageProcessorProtocol, PredictorProtocol
from app.core.config import get_settings
from app.deps import get_image_processor, get_predictor
from app.utils.file_validation import is_image_content_type

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    return HealthResponse(message="BANANA IS READY TO BE EATEN")


@router.post("/api/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    predictor: PredictorProtocol = Depends(get_predictor),
    image_processor: ImageProcessorProtocol = Depends(get_image_processor),
) -> PredictionResponse:
    settings = get_settings()

    if not is_image_content_type(file.content_type):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    contents = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="Upload exceeds size limit")

    try:
        img = image_processor.decode_to_bgr(contents)
        result = predictor.predict_from_bgr(img)
        return PredictionResponse(
            prediction=result.label,
            label=result.label,
            class_id=result.class_id,
            confidence=result.confidence,
        )
    except (InvalidImageError, ImageDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail="Processing error") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Processing error") from exc
