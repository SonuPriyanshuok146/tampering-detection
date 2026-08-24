import io
import base64
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

from tampering_detection.model import build_model
from tampering_detection.preprocessing import compute_ela
from tampering_detection.gradcam import generate_gradcam
from torchvision import transforms

app = FastAPI(title="Tampering Detection API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model()
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    temp_path = f"static/uploads/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    ela_img = compute_ela(temp_path)
    input_tensor = transform(ela_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_label = int(probs.argmax())
        confidence = float(probs[pred_label])

    overlay = generate_gradcam(model, input_tensor, device=device)
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "prediction": "tampered" if pred_label == 1 else "authentic",
        "confidence": round(confidence, 4),
        "gradcam_overlay_base64": overlay_b64,
    })