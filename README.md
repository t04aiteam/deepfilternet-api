# DeepFilterNet Speech Enhancement API

A FastAPI REST service that wraps [**DeepFilterNet**](https://github.com/Rikorose/DeepFilterNet)
— a low-complexity, real-time speech-enhancement (noise-suppression) model for
full-band 48 kHz audio. Upload a noisy audio file, get a denoised 48 kHz WAV back.

Backed by the official [`deepfilternet`](https://pypi.org/project/deepfilternet/)
package, which ships the pretrained **DeepFilterNet2 / DeepFilterNet3** weights —
there is no separate checkpoint to download.

## Prerequisites

- Python **3.9–3.11**
- **PyTorch 2.0.1 + torchaudio 2.0.2** (pinned — see note below)
- System library **libsndfile** (`apt-get install libsndfile1`; the Dockerfile handles this)
- ~300 MB disk for the model + dependencies

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. install the PINNED PyTorch (CPU build shown; use a CUDA index for GPU)
#    deepfilternet 0.5.x imports torchaudio.backend.common, which torchaudio
#    >=2.2 removed -- so these exact versions are required.
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 3. install the rest
pip install -r requirements.txt

# 4. configure
cp .env.example .env      # choose MODEL_NAME / POST_FILTER / limits
```

Note: the app does not load `.env` by itself. Docker Compose injects it via
`env_file:`; for local runs pass `--env-file .env` to uvicorn (shown below)
or export the variables in your shell.

## Environment variables

| Name                   | Required | Default          | Description                                              |
|------------------------|----------|------------------|----------------------------------------------------------|
| `MODEL_NAME`           | no       | `DeepFilterNet3` | `DeepFilterNet`, `DeepFilterNet2`, or `DeepFilterNet3`   |
| `POST_FILTER`          | no       | `false`          | Over-attenuate very noisy sections (`true`/`false`)      |
| `MAX_UPLOAD_BYTES`     | no       | `52428800`       | Reject uploads larger than this (50 MB default)          |
| `CORS_ORIGINS`         | no       | `*`              | Comma-separated allowed origins                          |
| `CUDA_VISIBLE_DEVICES` | no       | —                | The device is auto-detected (CUDA if visible, else CPU). Set to `""` to force CPU on a GPU machine |

## Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --env-file .env
```

(`python main.py` also works but serves on port **7551** and does not read `.env`.)

The model loads at startup, so the first request is not slowed by a cold load.
Interactive docs: http://localhost:8000/docs

## Run with Docker

```bash
cp .env.example .env
docker-compose up --build
```

## Endpoints

No authentication — the service is intended for trusted local-network deployment.

| Method | Path             | Body                      | Response                          |
|--------|------------------|---------------------------|-----------------------------------|
| GET    | `/health`        | —                         | JSON: status, model, device, sr   |
| GET    | `/api/v1/info`   | —                         | JSON: model metadata               |
| POST   | `/api/v1/enhance`| multipart file (`file=@`) | `audio/wav` (enhanced 48 kHz WAV)  |

Accepted uploads: `.wav`, `.flac`, `.ogg`, `.mp3` decode in memory; video (`mp4`/`mov`/`mkv`/`webm`, the audio track is used) and `m4a`/`aac` go through the host's `ffmpeg`. The file content decides, not its name. Output is always 48 kHz 16-bit WAV.

## Examples

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"DeepFilterNet3","device":"cpu","sample_rate":48000}
```

Model info:

```bash
curl http://localhost:8000/api/v1/info
# {"model":"DeepFilterNet3","device":"cpu","sample_rate":48000,"post_filter":false}
```

Enhance a noisy file (writes the denoised WAV to `enhanced.wav`):

```bash
curl -X POST http://localhost:8000/api/v1/enhance \
  -F "file=@noisy_audio.wav" \
  --output enhanced.wav
```

## Notes

- **Sample rate:** DeepFilterNet operates at 48 kHz. Uploads at other rates are
  resampled automatically; the response is always 48 kHz.
- **CPU vs GPU:** runs comfortably on CPU (it is designed for real-time use). A GPU
  is used automatically when one is visible and a CUDA torch build is installed;
  set `CUDA_VISIBLE_DEVICES=""` to force CPU.
- **Model choice:** `DeepFilterNet3` is newest and recommended. `DeepFilterNet2`
  is a solid, slightly lighter alternative.
- **Concurrency:** a single uvicorn worker serves requests sequentially through a
  thread-pool executor. For higher throughput run more workers/replicas (each
  loads its own model copy).

## License

DeepFilterNet is dual-licensed MIT / Apache-2.0. See the
[upstream repository](https://github.com/Rikorose/DeepFilterNet) for details.
