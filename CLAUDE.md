# FlickRec

## What this is
A content ranking API that scores and ranks movies for a given user using a multi-task MLP trained on MovieLens 1M. Built as a portfolio project demonstrating MLOps skills.

## Architecture
- Multi-task MLP in PyTorch: shared layers + p(like) and p(dislike) heads
- Score = p(like) - λ * p(dislike), λ=0.5 default
- Model weights stored in GCP Cloud Storage
- API served via FastAPI on GCP Cloud Run
- Docker image contains API code only, no weights
- GitHub Actions: test → build → push to Artifact Registry → deploy to Cloud Run

## Input features
- User embedding (32 dims)
- Movie embedding (32 dims)
- Genre multi-hot (18 dims)
- User metadata: age bucket, gender, occupation
- Total input dim: 85

## API
- POST /rank: takes user_id, samples 500 candidates, scores, returns top 20
- GET /health: health check
- GET /: simple UI

## Project phases
- [ ] Phase 1: Local setup
- [ ] Phase 2: GCP setup
- [ ] Phase 3: Training pipeline (train.py)
- [ ] Phase 4: API (api.py)
- [ ] Phase 5: Containerization (Dockerfile)
- [ ] Phase 6: CI/CD (GitHub Actions)
- [ ] Phase 7: Docs (README.md, DESIGN.md)

## Status
Phase 1 in progress.
