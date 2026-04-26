from __future__ import annotations


class BananaAppError(Exception):
    """Base error for the application."""


class InvalidImageError(BananaAppError):
    """Raised when the uploaded file is not a valid image."""


class ImageDecodeError(BananaAppError):
    """Raised when OpenCV fails to decode the image bytes."""


class PredictionError(BananaAppError):
    """Raised when model prediction fails."""
