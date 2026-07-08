FROM python:3.11-slim

# libsndfile1 is required by soundfile; git is occasionally needed by deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install pinned CPU torch first (smaller image). Override for GPU as needed.
# Pins required: deepfilternet 0.5.x needs torchaudio.backend.common (removed in >=2.2).
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
