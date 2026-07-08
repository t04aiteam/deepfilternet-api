"""Speech-enhancement endpoints.

POST /api/v1/enhance       -> upload a noisy audio file, get an enhanced WAV back
GET  /api/v1/info          -> model metadata

Audio is decoded with libsndfile (via soundfile), resampled to the model's
48 kHz rate with torchaudio's pure-tensor resampler, denoised, and streamed back
as a 48 kHz 16-bit WAV. Decoding via libsndfile (rather than torchaudio's file
IO) avoids any dependency on a system ffmpeg/sox backend for MP3/OGG.

No authentication: this service is intended for trusted local-network use.
"""

import asyncio
import io
import os

import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from models.loader import enhance_waveform, get_sample_rate
from schemas.request import InfoResponse

router = APIRouter()

# libsndfile (bundled with soundfile >=0.12) decodes all of these.
ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB


def _run_enhance(raw: bytes) -> bytes:
    """Blocking helper: decode -> resample -> enhance -> encode to WAV bytes.

    Runs in a thread-pool executor so it never blocks the event loop.
    """
    import torch
    from torchaudio.functional import resample

    target_sr = get_sample_rate()

    # Decode straight from memory with libsndfile. always_2d -> (samples, channels).
    data, orig_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)

    # DeepFilterNet expects (channels, samples) at the model sample rate.
    audio = torch.from_numpy(data.T.copy())
    if orig_sr != target_sr:
        audio = resample(audio, orig_sr, target_sr)

    enhanced = enhance_waveform(audio.numpy())  # (channels, samples)

    out = io.BytesIO()
    # soundfile expects (samples, channels); transpose back.
    sf.write(out, enhanced.T, target_sr, format="WAV", subtype="PCM_16")
    out.seek(0)
    return out.read()


@router.post("/enhance")
async def enhance_audio(file: UploadFile = File(...)):
    """Denoise an uploaded audio file and return an enhanced 48 kHz WAV.

    The response body is the raw WAV (Content-Type: audio/wav).
    """
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)} bytes > {MAX_BYTES} byte limit).",
        )

    try:
        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, _run_enhance, raw)
    except HTTPException:
        raise
    except Exception as e:  # decode / inference failure
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {e}")

    out_name = f"enhanced_{os.path.splitext(file.filename or 'audio')[0]}.wav"
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@router.get("/info", response_model=InfoResponse)
async def info():
    """Return model metadata (name, device, sample rate, post-filter flag)."""
    from models.loader import MODEL_NAME, POST_FILTER, get_device

    return InfoResponse(
        model=MODEL_NAME,
        device=get_device(),
        sample_rate=get_sample_rate(),
        post_filter=POST_FILTER,
    )