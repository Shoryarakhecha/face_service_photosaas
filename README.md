# PhotoSaaS Face Recognition Service

A standalone Python microservice using **InsightFace** to detect faces and
generate embeddings for face-matching search. Deployed separately from the
main Next.js app, since it needs Python + native ML libraries that don't run
on Vercel.

## What it does

1. `POST /extract-embeddings` — given a photo URL (e.g. a Cloudinary URL),
   detects every face in the image and returns a 512-dimensional embedding
   vector per face.
2. `POST /compare` — given one selfie embedding and a batch of photo
   embeddings, returns which photos contain a matching face (cosine
   similarity above a threshold).

This service is **stateless** — it does zero database work. Next.js owns all
storage; this service only does the ML computation.

## Local development

```bash
cd face-service
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

First run will download the InsightFace model weights (~300MB) — this only
happens once, cached locally afterward.

Test it's working:
```bash
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true}
```

## Deploy to Render (free tier)

1. Push this `face-service` folder to its own GitHub repo (or a subfolder of
   your existing repo — Render lets you point to a subdirectory).
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your repo. Render will detect `render.yaml` automatically.
4. If deploying from a subfolder, set **Root Directory** to `face-service`
5. Click **Create Web Service** — first deploy takes 5-10 minutes (Docker
   build + downloading ML dependencies).
6. Once live, copy your service URL (e.g.
   `https://photosaas-face-service.onrender.com`)

## Important: free tier behavior

Render's free tier **sleeps after 15 minutes of no traffic**. The next
request after sleep takes 30-90 seconds (container cold start + model
loading). This is expected — the Next.js side is built to show a "warming
up" message rather than timing out silently.

## Environment variables (set in Render dashboard)

| Variable | Value | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | Your Vercel domain, e.g. `https://yourapp.vercel.app` | Restricts CORS once you're in production (defaults to `*` for development) |

## API reference

### `POST /extract-embeddings`
```json
// Request
{ "image_url": "https://res.cloudinary.com/.../photo.jpg" }

// Response
{
  "faces": [
    {
      "embedding": [0.0123, -0.045, ...],  // 512 floats
      "bbox": [120.5, 80.2, 340.1, 310.8],
      "confidence": 0.97
    }
  ]
}
```

### `POST /compare`
```json
// Request
{
  "selfie_embedding": [0.0123, -0.045, ...],
  "candidates": {
    "photo_id_1": [[0.011, ...], [0.022, ...]],
    "photo_id_2": [[0.033, ...]]
  },
  "threshold": 0.45
}

// Response
{
  "matched_photo_ids": ["photo_id_1"],
  "scores": { "photo_id_1": 0.6234 }
}
```
# face_service_photosaas
