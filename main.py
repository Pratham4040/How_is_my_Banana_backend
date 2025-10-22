import io
import os
import logging
from typing import List
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
logger = logging.getLogger("BAnana")

class_names = ['0', '2-1', '4-3', '5-6', '7-8']

MODEL_DIR = os.path.join(os.path.dirname(__file__), "Model")
model_path = os.path.join(MODEL_DIR, "bananaV1.h5")
loaded_model = tf.keras.models.load_model(model_path)

def model_predict_from_array(img_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(img_rgb, (256, 256))
    scaled = resized / 255.0
    preds = loaded_model.predict(np.expand_dims(scaled, 0))
    return class_names[int(np.argmax(preds))]

app = FastAPI(
    title="How is my BANANA",
    description="Banana Ripeness Predictor",
    version=os.getenv("APP_VERSION", "1.0.0"),
)

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://howismybanana.netlify.app/", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "BANANA IS READY TO BE EATEN"}

@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        if not (file.content_type and file.content_type.startswith("image/")):
            raise HTTPException(status_code=400, detail="Only image uploads are supported")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        result = model_predict_from_array(img)
        return JSONResponse({"prediction": result})
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error processing image")
        raise HTTPException(status_code=500, detail="Processing error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)