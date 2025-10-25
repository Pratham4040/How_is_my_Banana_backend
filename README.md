# 🍌 How Is My Banana – Backend (FastAPI)

A lightweight FastAPI service that predicts banana ripeness from an uploaded image using a TensorFlow model (`bananaV1.h5`).

- Framework: FastAPI + Uvicorn
- ML: TensorFlow / Keras (.h5 model)
- Image I/O: OpenCV, NumPy
- Endpoint: `POST /api/predict`

##  Features
- Predicts ripeness class from an image
- Safe image validation + decoding with OpenCV
- CORS enabled for your frontend (localhost + Netlify)
- Production-friendly HOST/PORT via environment variables
- CPU-only safe on platforms without GPU (Render free tier)

##  Project Structure
```
backend/
├─ main.py                # FastAPI app + model loading + endpoints
├─ requirements.txt       # Python dependencies
├─ Model/
│  └─ bananaV1.h5        # Trained Keras model (kept in repo)
└─ __pycache__/          # (ignored)
```

##  Run Locally (Windows PowerShell)

```powershell
# 1) Go to backend folder
cd "c:\codes\PROJECTS - ML\HOW_IS_MY_BANANA\backend"

# 2) Create & activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Start the server (reads HOST/PORT from env if provided)
python .\main.py
# or (equivalent)
uvicorn main:app --host 0.0.0.0 --port 8000
```

By default, the API listens on `http://0.0.0.0:8000`.

##  Environment Variables
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `APP_VERSION` (default: `1.0.0`)
- Optional (recommended on GPU-less hosts): `CUDA_VISIBLE_DEVICES=-1`

##  API

### Health
```
GET /
```
Response
```json
{"message": "BANANA IS READY TO BE EATEN"}
```

### Predict
```
POST /api/predict
Content-Type: multipart/form-data
Form field: file=<image>
```
Response
```json
{"prediction": "4-3"}
```
Where prediction is one of: `"0"`, `"2-1"`, `"4-3"`, `"5-6"`, `"7-8"`.

#### curl example
```bash
curl -X POST \
  -F "file=@./test.jpg" \
  http://localhost:8000/api/predict
```

#### Python example
```python
import requests

url = "http://localhost:8000/api/predict"
with open("test.jpg", "rb") as f:
    r = requests.post(url, files={"file": ("test.jpg", f, "image/jpeg")})
print(r.status_code, r.json())
```

## 🧠 Model Notes
- The service loads `Model/bananaV1.h5` at startup.
- Images are converted BGR→RGB, resized to 256×256, scaled to [0,1], and predicted via `model.predict`.
- Keep the model file in the same `Model` folder in production.

## 🌐 CORS
CORS is configured in `main.py` and includes development and your deployed frontend domains. If you deploy the frontend to a new domain, add it to the `allow_origins` list.

## ☁️ Deploying to Render (Web Service)
- Root Directory: `backend` (recommended)
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment:
  - `PYTHON_VERSION` (e.g., `3.11.x`)
  - `CUDA_VISIBLE_DEVICES=-1` (prevents TensorFlow from probing GPU)

> Tip: On free tier, services sleep after 15 min. First request after idle may take ~30s (cold start).

##  Troubleshooting

- CUDA error like `failed call to cuInit` on Render:
  - Set env var `CUDA_VISIBLE_DEVICES=-1`
  - In `main.py`, we also guard with:
    ```python
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # and optionally
    # tf.config.set_visible_devices([], 'GPU')
    ```
- CORS error in browser:
  - Ensure the frontend origin is listed in `allow_origins` in `main.py`.
- 404 on `/api/predict`:
  - Use `POST` with `multipart/form-data` and field name `file`.
- 500 Processing error:
  - Ensure the image is a valid JPEG/PNG. The server logs will include the stack trace.

## 📜 License
This project’s code is provided as-is for educational/demo purposes. Add your preferred license here.

##  Acknowledgements
- FastAPI (https://fastapi.tiangolo.com)
- TensorFlow (https://www.tensorflow.org)
- OpenCV (https://opencv.org)
