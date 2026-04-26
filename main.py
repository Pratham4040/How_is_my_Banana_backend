from __future__ import annotations

import uvicorn

from app.core.config import get_settings
from app.main import create_app

app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)