# FlickRec

A movie ranking API. Given a user ID, it samples 500 movies the user hasn't
rated, scores them with a trained neural net, and returns the top 20. Trained on
MovieLens 1M, served with FastAPI, deployed to Cloud Run.

## Architecture

**Model** — a multi-task MLP (PyTorch). Inputs are an 85-dim vector: learned
user embedding (32) + movie embedding (32) + genre multi-hot (18) + user
metadata (age bucket, gender, occupation). A shared trunk (85 → 64 → 32, ReLU)
feeds two sigmoid heads: `p(like)` and `p(dislike)`. Ranking score is
`p(like) - 0.5 * p(dislike)`.

Labels come from MovieLens ratings: 4–5 → like, 1 → dislike, 2–3 → neutral
(both heads 0). The neutral class keeps the two heads from collapsing into
mirror images.

**API** — FastAPI app (`api.py`):
- `GET /` — minimal web UI
- `POST /rank` — `{"user_id": int}` → top 20 ranked movies
- `GET /user/{user_id}/genre_profile` — liked-genre counts for the user
- `GET /health` — health check

**Infrastructure** — trained weights live in GCS (`gs://flickrec-models`). The
Docker image contains API code only; it downloads `model.pt` from GCS on
startup. CI/CD (GitHub Actions) builds the image, pushes it to Artifact
Registry, and deploys to Cloud Run on every push to `main`.

## Run locally

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download MovieLens 1M, train, and write models/model.pt
python train.py

# Serve (loads model.pt from GCS, falling back to the local file)
uvicorn api:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080.

The GCS upload at the end of `train.py` is skipped with a warning if no
credentials are present; training and the local `model.pt` are unaffected.

## Run with Docker

```bash
docker build -t flickrec .
docker run -p 8080:8080 \
  -e GCS_BUCKET=flickrec-models \
  flickrec
```

The image ships without weights, so the container needs access to `model.pt` in
GCS at startup: provide application default credentials (e.g. mount a service
account key and set `GOOGLE_APPLICATION_CREDENTIALS`) and make sure the model
has been uploaded to the bucket.

## Project structure

```
api.py                       FastAPI service: loads the model, serves ranking + UI
train.py                     Training pipeline: download data, train, save model.pt, upload to GCS
requirements.txt             Pinned dependencies
Dockerfile                   Container build (CPU-only torch)
.dockerignore                Build context exclusions
.github/workflows/deploy.yml CI/CD: build, push to Artifact Registry, deploy to Cloud Run
CLAUDE.md                    Project notes and conventions
data/                        MovieLens raw data (gitignored)
models/                      Local model artifacts (gitignored)
tests/                       Tests
```
