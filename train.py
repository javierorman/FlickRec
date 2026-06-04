"""Train the FlickRec multi-task MLP on MovieLens 1M.

Pipeline: download data -> build labels and features -> train -> save model.pt
-> upload to GCS. Everything lives in this one file on purpose; see CLAUDE.md.
"""

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from google.cloud import storage

# --- Config ---
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
ML_DIR = os.path.join(DATA_DIR, "ml-1m")
MODEL_PATH = os.path.join("models", "model.pt")
GCS_BUCKET = "flickrec-models"
GCS_BLOB = "model.pt"

EMB_DIM = 32
EPOCHS = 10
BATCH_SIZE = 1024
LR = 1e-3
LAMBDA = 0.5

# The 18 MovieLens genres, in a fixed order so the multi-hot index is stable.
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# MovieLens encodes age as the lower bound of a bucket; map each to an index.
AGE_BUCKETS = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}


def download_data():
    """Download and unzip MovieLens 1M into data/ if not already present."""
    if os.path.isdir(ML_DIR):
        print(f"Dataset already present at {ML_DIR}")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-1m.zip")
    print(f"Downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"Extracted to {ML_DIR}")


def load_raw():
    """Read the three MovieLens .dat files into DataFrames."""
    read = lambda name, cols: pd.read_csv(
        os.path.join(ML_DIR, name),
        sep="::", engine="python", encoding="latin-1",
        header=None, names=cols,
    )
    ratings = read("ratings.dat", ["user_id", "movie_id", "rating", "ts"])
    users = read("users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    movies = read("movies.dat", ["movie_id", "title", "genres"])
    return ratings, users, movies


def build_lookups(users, movies):
    """Build the id->index maps and the per-id metadata/genre/title lookups."""
    # Embedding indices cover every user and movie so the API can score anyone.
    user2idx = {uid: i for i, uid in enumerate(users["user_id"])}
    movie2idx = {mid: i for i, mid in enumerate(movies["movie_id"])}

    genre_index = {g: i for i, g in enumerate(GENRES)}
    movie_genres = {}
    for mid, genres in zip(movies["movie_id"], movies["genres"]):
        vec = np.zeros(len(GENRES), dtype=np.float32)
        for g in genres.split("|"):
            if g in genre_index:
                vec[genre_index[g]] = 1.0
        movie_genres[mid] = vec

    movie_titles = dict(zip(movies["movie_id"], movies["title"]))

    user_features = {}
    for uid, gender, age, occ in zip(
        users["user_id"], users["gender"], users["age"], users["occupation"]
    ):
        age_bucket = AGE_BUCKETS[age]
        gender_val = 1 if gender == "M" else 0
        user_features[uid] = [age_bucket, gender_val, int(occ)]

    return user2idx, movie2idx, movie_genres, movie_titles, user_features


def build_examples(ratings, user2idx, movie2idx, movie_genres, user_features):
    """Filter ratings to like/dislike and turn them into model-ready tensors."""
    # like = rating 4-5, dislike = rating 1; drop the ambiguous 2-3 ratings.
    keep = ratings[ratings["rating"].isin([1, 4, 5])].copy()

    user_idx = keep["user_id"].map(user2idx).to_numpy()
    movie_idx = keep["movie_id"].map(movie2idx).to_numpy()
    genre = np.stack(keep["movie_id"].map(movie_genres).to_numpy())
    meta = np.stack(keep["user_id"].map(user_features).to_numpy()).astype(np.float32)

    like = (keep["rating"] >= 4).to_numpy().astype(np.float32)
    dislike = (keep["rating"] == 1).to_numpy().astype(np.float32)

    return (
        torch.tensor(user_idx, dtype=torch.long),
        torch.tensor(movie_idx, dtype=torch.long),
        torch.tensor(genre, dtype=torch.float32),
        torch.tensor(meta, dtype=torch.float32),
        torch.tensor(like, dtype=torch.float32).unsqueeze(1),
        torch.tensor(dislike, dtype=torch.float32).unsqueeze(1),
    )


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


def train_model(tensors, n_users, n_movies, device):
    """Train the MLP with an 80/20 split and return the trained model."""
    train_parts, val_parts = train_test_split(
        list(range(len(tensors[0]))), test_size=0.2, random_state=42
    )
    # train_test_split on indices keeps the split reproducible and memory-light.
    train_ds = TensorDataset(*[t[train_parts] for t in tensors])
    val_ds = TensorDataset(*[t[val_parts] for t in tensors])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MultiTaskMLP(n_users, n_movies).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for u, m, g, meta, like, dislike in train_loader:
            u, m, g, meta = u.to(device), m.to(device), g.to(device), meta.to(device)
            like, dislike = like.to(device), dislike.to(device)
            optimizer.zero_grad()
            p_like, p_dislike = model(u, m, g, meta)
            loss = bce(p_like, like) + bce(p_dislike, dislike)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(u)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for u, m, g, meta, like, dislike in val_loader:
                u, m, g, meta = u.to(device), m.to(device), g.to(device), meta.to(device)
                like, dislike = like.to(device), dislike.to(device)
                p_like, p_dislike = model(u, m, g, meta)
                loss = bce(p_like, like) + bce(p_dislike, dislike)
                val_loss += loss.item() * len(u)

        print(
            f"Epoch {epoch:2d}/{EPOCHS}  "
            f"train_loss={train_loss / len(train_ds):.4f}  "
            f"val_loss={val_loss / len(val_ds):.4f}"
        )

    return model


def save_model(model, lookups, n_users, n_movies):
    """Save weights plus everything the API needs to reconstruct features."""
    user2idx, movie2idx, movie_genres, movie_titles, user_features = lookups
    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "movie2idx": movie2idx,
            "movie_genres": movie_genres,
            "movie_titles": movie_titles,
            "user_features": user_features,
            "n_users": n_users,
            "n_movies": n_movies,
            "lambda_": LAMBDA,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


def upload_to_gcs():
    """Upload model.pt to GCS using application default credentials."""
    client = storage.Client()
    blob = client.bucket(GCS_BUCKET).blob(GCS_BLOB)
    blob.upload_from_filename(MODEL_PATH)
    print(f"Uploaded {MODEL_PATH} to gs://{GCS_BUCKET}/{GCS_BLOB}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    download_data()
    ratings, users, movies = load_raw()
    lookups = build_lookups(users, movies)
    user2idx, movie2idx, movie_genres, _, user_features = lookups
    n_users, n_movies = len(user2idx), len(movie2idx)
    print(f"{n_users} users, {n_movies} movies")

    tensors = build_examples(ratings, user2idx, movie2idx, movie_genres, user_features)
    print(f"{len(tensors[0])} training examples after filtering")

    model = train_model(tensors, n_users, n_movies, device)
    save_model(model, lookups, n_users, n_movies)
    upload_to_gcs()


if __name__ == "__main__":
    main()
