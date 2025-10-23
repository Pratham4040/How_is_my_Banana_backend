import io
import os
import logging
from typing import List
import numpy as np
import cv2

# Force TensorFlow to use CPU only (no GPU on Render)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

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
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:10000",
        "https://howismybanana.netlify.app",
        "https://how-is-my-banana-backend.onrender.com"
    ],
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