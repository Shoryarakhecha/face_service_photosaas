# Dockerfile
FROM python:3.10-slim

# System dependencies needed by opencv-python-headless and onnxruntime,
# plus build tools since insightface compiles a C++ extension at install time.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT