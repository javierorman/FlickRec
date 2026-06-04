# FlickRec

FlickRec is a movie recommendation API that ranks movies for a given user with a neural net trained on MovieLens 1M.

## Live demo

https://flickrec-zyqqaqoc2a-uc.a.run.app

## How it works

- **Dataset:** MovieLens 1M (1M ratings, 6040 users, 3883 movies)
- **Model:** a multi-task MLP with shared layers and two heads: p(like) and p(dislike)

### Model

- Multi-task MLP: shared layers feed two heads, p(like) and p(dislike)
- Labels: like=4-5, dislike=1, neutral=2-3. Neutral examples are needed: without them the two heads learn to be mirror images and scores saturate near 1.0
- Score: p(like) - 0.5 * p(dislike)
- Features: user embedding, movie embedding, genre multi-hot, age bucket, gender, occupation

### Candidate generation

Each request samples 500 unrated movies at random, scores them, and returns the top 20. In production this would be a retrieval stage with a two-tower model and ANN search.

### Known limitations

- Scores are not calibrated probabilities; they reflect ranking order
- No session context or recency signals
- Candidate generation is random, not learned retrieval

## Run locally

```bash
python train.py              # train model and upload weights to GCS
uvicorn api:app --port 8080  # serve the API
```

## Run with Docker

```bash
docker build -t flickrec .
docker run -p 8080:8080 flickrec
```

## Stack

Python, PyTorch, FastAPI, Docker, GCP Cloud Run, GCP Cloud Storage, GitHub Actions
