FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Install the CPU-only torch first so we don't pull the ~3.5GB CUDA stack;
# Cloud Run has no GPU. The second install then sees torch already satisfied.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY api.py .

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
