import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from backend.model import analyze_image, metadata
except ImportError:  # pragma: no cover - fallback for direct backend execution
    try:
        from .model import analyze_image, metadata
    except ImportError:  # pragma: no cover - legacy direct-module execution
        from model import analyze_image, metadata

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="TamperVision-Lite API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")

ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_BYTES = 15 * 1024 * 1024


def save_image(path: Path, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Could not save {path.name}")


@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded", "project": metadata.get("project", "TamperVision-Lite")}


@app.get("/metadata")
def get_metadata():
    return metadata


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED:
        raise HTTPException(status_code=415, detail="Unsupported image type.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 15 MB.")

    try:
        result = analyze_image(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    original_name = Path(file.filename or "image").stem.replace(" ", "_")[:80]

    mask_name = f"{job_id}_pseudo_mask.png"
    loc_name = f"{job_id}_localization.png"
    cam_name = f"{job_id}_gradcam.png"
    overlay_name = f"{job_id}_mask_overlay.png"
    final_name = f"{job_id}_final.png"

    save_image(RESULT_DIR / mask_name, cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2RGB))
    save_image(RESULT_DIR / loc_name, result["localized"])
    save_image(RESULT_DIR / cam_name, result["gradcam"])
    save_image(RESULT_DIR / overlay_name, result["pseudo_overlay"])

    # Four-panel final visualization.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(result["image"]); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(result["gradcam"]); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(result["mask"], cmap="gray"); axes[2].set_title("Pseudo-Mask"); axes[2].axis("off")
    axes[3].imshow(result["localized"]); axes[3].set_title("Bounding Boxes"); axes[3].axis("off")
    fig.suptitle(f'{result["prediction"]} | Probability: {result["probability"]:.2%}', fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / final_name, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "prediction": result["prediction"],
        "probability": round(result["probability"], 6),
        "probability_percent": round(result["probability"] * 100, 2),
        "threshold": result["threshold"],
        "regions": result["regions"],
        "localization_note": "Grad-CAM-based approximate localization; mask is a derived pseudo-mask, not ground-truth segmentation.",
        "images": {
            "gradcam": f"/results/{cam_name}",
            "mask": f"/results/{mask_name}",
            "mask_overlay": f"/results/{overlay_name}",
            "localization": f"/results/{loc_name}",
            "final": f"/results/{final_name}",
        },
    }
