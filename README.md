# FlickRec

FlickRec is a movie recommendation API that ranks movies for a given user with a neural net trained on MovieLens 1M.

## Live demo

https://flickrec-zyqqaqoc2a-uc.a.run.app

## How it works

- **Dataset:** MovieLens 1M (1M ratings, 6040 users, 3883 movies)
- **Model:** a multi-task MLP with shared layers and two heads: p(like) and p(dislike)
- **Scoring:** score = p(like) - 0.5 * p(dislike)
- **API:** given a user, it samples 500 unrated candidate movies, scores each, and returns the top 20

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
