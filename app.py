import io
import os
import uuid
import base64
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from tampering_detection.model import build_model
from tampering_detection.preprocessing import compute_ela
from tampering_detection.gradcam import generate_gradcam
from torchvision import transforms

app = FastAPI(title="Tampering Detection API")

# --- CORS, configured via an environment variable ---
# Locally (no env var set) this defaults to "*" so nothing breaks during
# development. On Render, set ALLOWED_ORIGIN to your real Netlify URL via
# the dashboard -- no code change or redeploy needed if your frontend URL
# ever changes, just update the env var and Render restarts automatically.
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/uploads"
CHECKPOINT_PATH = "checkpoints/best_model.pth"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_MB = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure the upload folder exists, even on a fresh clone of the project
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model once at startup. Fail loudly and clearly if the checkpoint
# is missing or invalid, instead of crashing later with a confusing traceback
# the first time someone hits /predict.
if not os.path.exists(CHECKPOINT_PATH):
    raise RuntimeError(
        f"Model checkpoint not found at '{CHECKPOINT_PATH}'. "
        "Run training first: python -m tampering_detection.train"
    )

model = build_model()
try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
except Exception as e:
    raise RuntimeError(
        f"Failed to load checkpoint at '{CHECKPOINT_PATH}'. It may be empty "
        f"or corrupted from an interrupted training run. Re-run training to "
        f"regenerate it. Original error: {e}"
    )

model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


@app.get("/")
async def root():
    return {"status": "ok", "message": "Tampering Detection API is running. See /docs for usage."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- Validate content type ---
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPG or PNG image.",
        )

    # --- Read and validate size ---
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File is larger than {MAX_FILE_SIZE_MB}MB. Please upload a smaller image.",
        )

    # --- Save to a unique temp path so concurrent/duplicate filenames never collide ---
    safe_ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    if safe_ext not in (".jpg", ".jpeg", ".png"):
        safe_ext = ".jpg"
    temp_filename = f"{uuid.uuid4().hex}{safe_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(contents)

        # --- Verify it's actually a readable image before running ELA ---
        try:
            ela_img = compute_ela(temp_path)
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Could not read this file as an image. It may be corrupted or not a real image.",
            )

        input_tensor = transform(ela_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_label = int(probs.argmax())
            confidence = float(probs[pred_label])
            authentic_prob = float(probs[0])   # Au = label 0
            tampered_prob = float(probs[1])    # Tp = label 1

        try:
            overlay = generate_gradcam(model, input_tensor, device=device)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_b64 = base64.b64encode(buffer).decode("utf-8")
        except Exception:
            # If Grad-CAM fails for any reason, still return the prediction
            # rather than failing the whole request.
            overlay_b64 = None

        return JSONResponse({
            "prediction": "tampered" if pred_label == 1 else "authentic",
            "confidence": round(confidence, 4),
            "probabilities": {
                "authentic": round(authentic_prob, 4),
                "tampered": round(tampered_prob, 4),
            },
            "gradcam_overlay_base64": overlay_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    finally:
        # Clean up the temp upload regardless of success or failure,
        # so static/uploads/ doesn't fill up over time.
        if os.path.exists(temp_path):
            os.remove(temp_path)