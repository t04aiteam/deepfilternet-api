FROM python:3.11-slim

# libsndfile1 is required by soundfile; ffmpeg decodes video and m4a uploads.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Pinned CUDA 11.8 torch, as deployed on c09 (HDSD manual: device cuda). For a
# CPU-only image use --index-url https://download.pytorch.org/whl/cpu and remove
# the GPU block in docker-compose.yml.
# Pins required: deepfilternet 0.5.x needs torchaudio.backend.common (removed in >=2.2).
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7551
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7551", "--workers", "1"]
