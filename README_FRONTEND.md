# TamperVision-Lite — Web Application

This package connects the supplied trained Keras model and metadata to a browser frontend through a FastAPI backend.

## Architecture

Browser frontend → FastAPI `/predict` → `.keras` model → prediction + Grad-CAM → pseudo-mask + bounding boxes → result images → browser.

The `.pkl` file is metadata/configuration. The `.keras` file is the actual trained neural network.

## Files added

- `backend/model.py` — loads the exact supplied `.keras` model and `.pkl`, preprocessing, prediction, Grad-CAM, pseudo-mask and bounding-box localization.
- `backend/main.py` — FastAPI API and result-file serving.
- `frontend/index.html` — web UI.
- `frontend/style.css` — UI styling.
- `frontend/script.js` — upload/API/result handling.
- `backend/requirements.txt` — Python dependencies.

## Run locally

### 1. Create environment

Use Python 3.10 or 3.11 for the TensorFlow environment.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Start backend

Windows:
```bash
run_backend.bat
```

macOS/Linux:
```bash
./run_backend.sh
```

The API starts at `http://127.0.0.1:8000`.

### 4. Start frontend

From the project root, run a simple static server:

```bash
python -m http.server 5500 --directory frontend
```

Open:

`http://127.0.0.1:5500`

Do not open `index.html` directly with `file://` if the browser blocks API requests.

## API

- `GET /health` — backend/model health.
- `GET /metadata` — supplied project metadata.
- `POST /predict` — multipart image upload.

## Important research note

The localization is Grad-CAM-based approximate localization. The binary mask is a Grad-CAM-derived pseudo-mask, not a ground-truth pixel-level segmentation mask. The classifier was not trained with pixel-level segmentation supervision.

## Supplied model artifacts

The project includes the uploaded final `.keras` model and `.pkl` metadata in `backend/model/`.
