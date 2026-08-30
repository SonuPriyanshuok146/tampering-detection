# Slim Python base keeps the image smaller — important since Hugging Face
# Spaces free tier has limited storage and build time.
FROM python:3.10-slim

WORKDIR /app

# OpenCV needs these system libraries to work inside a minimal container —
# without them, "import cv2" fails at runtime with a cryptic .so error.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (before copying code) so Docker can
# cache this layer -- rebuilds are much faster if only your code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project (respecting .dockerignore)
COPY . .
RUN pip install -e .

# Hugging Face Spaces (Docker SDK) expects the container to listen on
# port 7860 specifically -- this is not optional, it's how HF routes traffic.
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
