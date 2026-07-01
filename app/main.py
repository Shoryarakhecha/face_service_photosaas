# app/main.py
import os
import time
import logging
from typing import List, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import insightface
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-service")

app = FastAPI(title="PhotoSaaS Face Recognition Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# MODEL LOADING (lazy — loads on first request, not at startup)
# Using buffalo_sc (small/compact) instead of buffalo_l to stay within
# Render free tier's 512MB RAM limit.
# ─────────────────────────────────────────
_face_app: Optional[FaceAnalysis] = None


def get_face_app() -> FaceAnalysis:
    """Load the model once and reuse it for all subsequent requests."""
    global _face_app
    if _face_app is None:
        logger.info("Loading InsightFace model (buffalo_sc)...")
        start = time.time()
        fa = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(320, 320))
        _face_app = fa
        logger.info(f"Model loaded in {time.time() - start:.1f}s")
    return _face_app


# ─────────────────────────────────────────
# REQUEST / RESPONSE SHAPES
# ─────────────────────────────────────────
class ExtractRequest(BaseModel):
    image_url: str


class FaceEmbedding(BaseModel):
    embedding: List[float]
    bbox: List[float]
    confidence: float


class ExtractResponse(BaseModel):
    faces: List[FaceEmbedding]


class CompareRequest(BaseModel):
    selfie_embedding: List[float]
    candidates: dict[str, List[List[float]]]
    threshold: float = 0.45


class CompareResponse(BaseModel):
    matched_photo_ids: List[str]
    scores: dict[str, float]


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def download_image(url: str) -> np.ndarray:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not download image: {e}")

    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _face_app is not None}


@app.post("/extract-embeddings", response_model=ExtractResponse)
def extract_embeddings(req: ExtractRequest):
    fa = get_face_app()
    img = download_image(req.image_url)
    faces = fa.get(img)

    results = [
        FaceEmbedding(
            embedding=face.normed_embedding.tolist(),
            bbox=face.bbox.tolist(),
            confidence=float(face.det_score),
        )
        for face in faces
    ]
    return ExtractResponse(faces=results)


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    selfie_vec = np.array(req.selfie_embedding)
    matched_ids = []
    scores: dict[str, float] = {}

    for photo_id, face_embeddings in req.candidates.items():
        best_score = 0.0
        for emb in face_embeddings:
            score = cosine_similarity(selfie_vec, np.array(emb))
            best_score = max(best_score, score)

        if best_score >= req.threshold:
            matched_ids.append(photo_id)
            scores[photo_id] = round(best_score, 4)

    matched_ids.sort(key=lambda pid: scores[pid], reverse=True)
    return CompareResponse(matched_photo_ids=matched_ids, scores=scores)