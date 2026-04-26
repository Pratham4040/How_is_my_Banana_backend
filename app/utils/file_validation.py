from __future__ import annotations


def is_image_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type.lower().startswith("image/")
