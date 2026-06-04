"""FlickRec ranking API.

Loads the trained multi-task MLP (from GCS, falling back to a local copy) and
serves rankings over HTTP. Everything lives in this one file on purpose; see
CLAUDE.md. The MultiTaskMLP class is copied verbatim from train.py so the API
has no dependency on the training code.
"""

import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from google.cloud import storage

# --- Config ---
GCS_BUCKET = "flickrec-models"
GCS_BLOB = "model.pt"
LOCAL_PATH = os.path.join("models", "model.pt")

EMB_DIM = 32


# --- Model (copied verbatim from train.py, no import from train.py) ---
class MultiTaskMLP(nn.Module):
    """Shared trunk over [user_emb | movie_emb | genre | meta] with two heads."""

    def __init__(self, n_users, n_movies):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, EMB_DIM)
        self.movie_emb = nn.Embedding(n_movies, EMB_DIM)
        # Input dim 85 = user 32 + movie 32 + genre 18 + meta 3.
        self.shared = nn.Sequential(
            nn.Linear(85, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.like_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.dislike_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, user_idx, movie_idx, genre, meta):
        x = torch.cat(
            [self.user_emb(user_idx), self.movie_emb(movie_idx), genre, meta],
            dim=1,
        )
        h = self.shared(x)
        return self.like_head(h), self.dislike_head(h)


def load_checkpoint():
    """Download model.pt from GCS; fall back to the local copy if that fails."""
    try:
        client = storage.Client()
        blob = client.bucket(GCS_BUCKET).blob(GCS_BLOB)
        tmp = os.path.join(tempfile.gettempdir(), "flickrec_model.pt")
        blob.download_to_filename(tmp)
        print(f"Loaded model from gs://{GCS_BUCKET}/{GCS_BLOB}", flush=True)
        return torch.load(tmp, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"GCS download failed ({e}); trying local {LOCAL_PATH}", flush=True)
        if os.path.exists(LOCAL_PATH):
            return torch.load(LOCAL_PATH, map_location="cpu", weights_only=False)
        raise RuntimeError("No model available from GCS or local disk")


# Load the model once at import time, before the app serves any request.
_ckpt = load_checkpoint()
user2idx = _ckpt["user2idx"]
movie2idx = _ckpt["movie2idx"]
movie_genres = _ckpt["movie_genres"]
movie_titles = _ckpt["movie_titles"]
user_features = _ckpt["user_features"]
user_rated = _ckpt["user_rated"]
lambda_ = _ckpt["lambda_"]

model = MultiTaskMLP(_ckpt["n_users"], _ckpt["n_movies"])
model.load_state_dict(_ckpt["model_state_dict"])
model.eval()

app = FastAPI(title="FlickRec")


class RankRequest(BaseModel):
    user_id: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rank")
def rank(req: RankRequest):
    """Sample 500 candidate movies, score them, and return the top 20."""
    user_id = req.user_id
    # Candidate pool is every movie the user has not already rated.
    rated = user_rated.get(user_id, set())
    pool = [m for m in movie2idx.keys() if m not in rated]
    n = min(500, len(pool))
    candidates = random.sample(pool, n)

    if user_id in user2idx:
        u_idx = user2idx[user_id]
        meta = user_features[user_id]
    else:
        # Unknown user: use embedding index 0 and neutral metadata so we can
        # still return something sensible rather than erroring out.
        u_idx = 0
        meta = [0, 0, 0]

    user_t = torch.tensor([u_idx] * n, dtype=torch.long)
    movie_t = torch.tensor([movie2idx[m] for m in candidates], dtype=torch.long)
    genre_t = torch.tensor(
        np.stack([movie_genres[m] for m in candidates]), dtype=torch.float32
    )
    meta_t = torch.tensor([meta] * n, dtype=torch.float32)

    with torch.no_grad():
        p_like, p_dislike = model(user_t, movie_t, genre_t, meta_t)
    p_like = p_like.squeeze(1)
    p_dislike = p_dislike.squeeze(1)
    score = p_like - lambda_ * p_dislike

    top = torch.argsort(score, descending=True)[:20].tolist()
    recommendations = []
    for rank_i, idx in enumerate(top, start=1):
        mid = candidates[idx]
        recommendations.append(
            {
                "rank": rank_i,
                "movie_id": int(mid),
                "title": movie_titles[mid],
                "p_like": round(float(p_like[idx]), 4),
                "p_dislike": round(float(p_dislike[idx]), 4),
                "score": round(float(score[idx]), 4),
            }
        )

    return {"user_id": user_id, "recommendations": recommendations}


# Simple inline UI. Built per request so the user dropdown reflects the model.
PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>FlickRec</title></head>
<body>
  <h1>FlickRec</h1>
  <label>User: <select id="user">__OPTIONS__</select></label>
  <button onclick="getRecs()">Get Recommendations</button>
  <table border="1" cellpadding="4" id="results">
    <thead>
      <tr><th>Rank</th><th>Title</th><th>p(like)</th><th>p(dislike)</th><th>Score</th></tr>
    </thead>
    <tbody></tbody>
  </table>
  <script>
    async function getRecs() {
      const userId = parseInt(document.getElementById('user').value);
      const res = await fetch('/rank', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user_id: userId}),
      });
      const data = await res.json();
      const body = document.querySelector('#results tbody');
      body.innerHTML = '';
      for (const r of data.recommendations) {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${r.rank}</td><td>${r.title}</td>` +
          `<td>${r.p_like}</td><td>${r.p_dislike}</td><td>${r.score}</td>`;
        body.appendChild(row);
      }
    }
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    # Populate the dropdown with the first 100 user IDs from the model.
    first_100 = list(user2idx.keys())[:100]
    options = "".join(f'<option value="{uid}">{uid}</option>' for uid in first_100)
    return PAGE_TEMPLATE.replace("__OPTIONS__", options)
