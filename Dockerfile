# Dockerfile
# InsightFace depends on OpenCV + ONNX Runtime, which need system libraries
# that Render's default Python buildpack doesn't reliably include. Docker
# gives us full control over the environment.

FROM python:3.10-slim

# System dependencies needed by opencv-python-headless and onnxruntime,
# plus build tools in case any package needs to compile from source.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Render injects $PORT at runtime — must bind to it, not a hardcoded port
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT