"""FlickRec ranking API: serves movie rankings from the trained model."""

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

GCS_BUCKET = os.environ.get("GCS_BUCKET", "flickrec-models")
GCS_BLOB = "model.pt"
LOCAL_PATH = os.path.join("models", "model.pt")

EMB_DIM = 32

# Same genre order as train.py so the multi-hot indices line up.
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# Copied verbatim from train.py so the API doesn't depend on the training code.
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
    """Download model.pt from GCS, falling back to the local copy."""
    try:
        tmp = os.path.join(tempfile.gettempdir(), "flickrec_model.pt")
        storage.Client().bucket(GCS_BUCKET).blob(GCS_BLOB).download_to_filename(tmp)
        print(f"Loaded model from gs://{GCS_BUCKET}/{GCS_BLOB}", flush=True)
        return torch.load(tmp, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"GCS unavailable ({e}); loading local {LOCAL_PATH}", flush=True)
        return torch.load(LOCAL_PATH, map_location="cpu", weights_only=False)


# Load the model once at import time, before the app serves any request.
_ckpt = load_checkpoint()
user2idx = _ckpt["user2idx"]
movie2idx = _ckpt["movie2idx"]
movie_genres = _ckpt["movie_genres"]
movie_titles = _ckpt["movie_titles"]
user_features = _ckpt["user_features"]
user_rated = _ckpt["user_rated"]
user_liked_genres = _ckpt["user_liked_genres"]
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
    rated = user_rated.get(user_id, set())
    pool = [m for m in movie2idx.keys() if m not in rated]
    n = min(500, len(pool))
    candidates = random.sample(pool, n)

    if user_id in user2idx:
        u_idx = user2idx[user_id]
        meta = user_features[user_id]
    else:
        # Unknown user: fall back to embedding 0 and neutral metadata.
        u_idx = 0
        meta = [0, 0, 0]

    movie_indices = [movie2idx[m] for m in candidates]
    genre_vectors = [movie_genres[m] for m in candidates]

    user_t = torch.tensor([u_idx] * n, dtype=torch.long)
    movie_t = torch.tensor(movie_indices, dtype=torch.long)
    genre_t = torch.tensor(np.stack(genre_vectors), dtype=torch.float32)
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
        genres = [GENRES[i] for i, on in enumerate(movie_genres[mid]) if on]
        recommendations.append(
            {
                "rank": rank_i,
                "movie_id": int(mid),
                "title": movie_titles[mid],
                "genres": ", ".join(genres[:3]),
                "p_like": round(float(p_like[idx]), 4),
                "p_dislike": round(float(p_dislike[idx]), 4),
                "score": round(float(score[idx]), 4),
            }
        )

    return {"user_id": user_id, "recommendations": recommendations}


@app.get("/user/{user_id}/genre_profile")
def genre_profile(user_id: int):
    """Return the genre counts of the movies this user liked (rating >= 4)."""
    return {"user_id": user_id, "genres": user_liked_genres.get(user_id, {})}


# Built per request so the dropdown can list the model's users.
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlickRec</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #0f0f0f;
      color: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header { padding: 24px 32px; border-bottom: 1px solid #2a2a2a; }
    h1 { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
    h1 .accent { color: #e50914; }
    .controls { display: flex; gap: 12px; align-items: center; padding: 24px 32px; }
    .controls label { color: #aaaaaa; font-size: 14px; }
    select {
      background: #1a1a1a; color: #ffffff; border: 1px solid #2a2a2a;
      padding: 10px 12px; border-radius: 6px; font-size: 14px; min-width: 120px;
    }
    select:focus { outline: none; border-color: #e50914; }
    button {
      background: #e50914; color: #ffffff; border: none; padding: 10px 18px;
      border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer;
    }
    button:hover { background: #f6121d; }
    button:disabled { opacity: 0.6; cursor: default; }
    .grid {
      display: grid; grid-template-columns: 360px 1fr; gap: 24px;
      padding: 0 32px 32px;
    }
    .panel {
      background: #1a1a1a; border: 1px solid #2a2a2a;
      border-radius: 10px; padding: 20px;
    }
    .panel h2 {
      margin: 0 0 16px; font-size: 13px; color: #aaaaaa; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.5px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th {
      text-align: left; color: #aaaaaa; font-weight: 600;
      padding: 10px 12px; border-bottom: 1px solid #2a2a2a;
    }
    td { padding: 10px 12px; }
    tbody tr:nth-child(odd) { background: #1a1a1a; }
    tbody tr:nth-child(even) { background: #222222; }
    tbody tr:hover { background: #2a2a2a; }
    .rank { color: #e50914; font-weight: 700; }
    .muted { color: #aaaaaa; }
    .genres { color: #aaaaaa; font-size: 12px; }
    .empty { color: #aaaaaa; font-size: 14px; }
  </style>
</head>
<body>
  <header><h1>Flick<span class="accent">Rec</span></h1></header>

  <div class="controls">
    <label for="user">User</label>
    <select id="user">__OPTIONS__</select>
    <button id="btn" onclick="getRecs()">Get Recommendations</button>
  </div>

  <div class="grid" id="grid" style="display:none;">
    <div class="panel">
      <h2>Liked Genres</h2>
      <div id="chart"><p class="empty">Select a user to see their genre profile.</p></div>
    </div>
    <div class="panel">
      <h2>Top 20 Recommendations</h2>
      <table>
        <thead>
          <tr><th>Rank</th><th>Title</th><th>Genres</th><th>Score</th><th>p(like)</th><th>p(dislike)</th></tr>
        </thead>
        <tbody id="recs"></tbody>
      </table>
    </div>
  </div>

  <script>
    const userSel = document.getElementById('user');
    const btn = document.getElementById('btn');
    const grid = document.getElementById('grid');

    // Draw the liked-genre counts as simple HTML bars.
    function drawChart(genres) {
      const container = document.getElementById('chart');
      const entries = Object.entries(genres).sort((a, b) => b[1] - a[1]);
      if (entries.length === 0) {
        container.innerHTML = '<p class="empty">No liked genres for this user.</p>';
        return;
      }
      const max = entries[0][1];
      let html = '';
      for (const [genre, count] of entries) {
        const pct = (count / max) * 100;
        html +=
          `<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:12px;">` +
          `<span style="width:90px; color:#aaaaaa;">${genre}</span>` +
          `<div style="background:#e50914; height:14px; border-radius:3px; width:${pct}%;"></div>` +
          `<span style="color:#ffffff;">${count}</span>` +
          `</div>`;
      }
      container.innerHTML = html;
    }

    async function loadGenreProfile(userId) {
      const res = await fetch(`/user/${userId}/genre_profile`);
      const data = await res.json();
      drawChart(data.genres);
    }

    async function getRecs() {
      const userId = parseInt(userSel.value);
      const original = btn.textContent;
      btn.disabled = true;
      btn.textContent = 'Loading...';
      grid.style.display = 'grid';
      try {
        await loadGenreProfile(userId);
        const res = await fetch('/rank', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({user_id: userId}),
        });
        const data = await res.json();
        const body = document.getElementById('recs');
        body.innerHTML = '';
        for (const r of data.recommendations) {
          const row = document.createElement('tr');
          row.innerHTML =
            `<td class="rank">${r.rank}</td>` +
            `<td>${r.title}</td>` +
            `<td class="genres">${r.genres}</td>` +
            `<td>${r.score}</td>` +
            `<td class="muted">${r.p_like}</td>` +
            `<td class="muted">${r.p_dislike}</td>`;
          body.appendChild(row);
        }
      } finally {
        btn.disabled = false;
        btn.textContent = original;
      }
    }

    // Refresh the genre profile as soon as a different user is selected.
    userSel.addEventListener('change', () => {
      grid.style.display = 'grid';
      loadGenreProfile(parseInt(userSel.value));
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    # First 100 users is enough to demo the dropdown.
    first_100 = list(user2idx.keys())[:100]
    options = "".join(f'<option value="{uid}">{uid}</option>' for uid in first_100)
    return PAGE_TEMPLATE.replace("__OPTIONS__", options)
