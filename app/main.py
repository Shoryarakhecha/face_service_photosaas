# app/main.py
# Face recognition microservice using InsightFace.
# Two endpoints:
#   POST /extract-embeddings  — given an image URL, detect all faces and
#                                return one embedding vector per face
#   POST /compare             — given one selfie embedding and a list of
#                                photo embeddings, return matches above a
#                                similarity threshold
#
# Stateless by design — Next.js owns all storage (Postgres). This service
# only does the ML work and never persists anything itself.

import os
import io
import time
import logging
from typing import List

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

# Allow requests from your Next.js app. In production, replace "*" with
# your actual Vercel domain to avoid the API being callable from anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# MODEL LOADING (happens once at startup — this is the slow part on cold start)
# ─────────────────────────────────────────
face_app: FaceAnalysis | None = None


@app.on_event("startup")
def load_model():
    global face_app
    logger.info("Loading InsightFace model (buffalo_l)...")
    start = time.time()
    # buffalo_l = the standard accurate model bundle (detection + recognition).
    # ctx_id=-1 forces CPU mode, since Render's free tier has no GPU.
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    logger.info(f"Model loaded in {time.time() - start:.1f}s")


# ─────────────────────────────────────────
# REQUEST / RESPONSE SHAPES
# ─────────────────────────────────────────
class ExtractRequest(BaseModel):
    image_url: str


class FaceEmbedding(BaseModel):
    embedding: List[float]
    bbox: List[float]  # [x1, y1, x2, y2] — useful later for face-crop thumbnails
    confidence: float


class ExtractResponse(BaseModel):
    faces: List[FaceEmbedding]


class CompareRequest(BaseModel):
    selfie_embedding: List[float]
    # photo_id -> list of face embeddings detected in that photo
    candidates: dict[str, List[List[float]]]
    threshold: float = 0.45  # cosine similarity threshold; tuned conservatively


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
    return {"status": "ok", "model_loaded": face_app is not None}


@app.post("/extract-embeddings", response_model=ExtractResponse)
def extract_embeddings(req: ExtractRequest):
    if face_app is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")

    img = download_image(req.image_url)
    faces = face_app.get(img)

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

    # Sort matches by score, best first
    matched_ids.sort(key=lambda pid: scores[pid], reverse=True)

    return CompareResponse(matched_photo_ids=matched_ids, scores=scores)
