"""Speech-enhancement endpoints.

POST /api/v1/enhance       -> upload a noisy audio file, get an enhanced WAV back
GET  /api/v1/info          -> model metadata

Audio is decoded with libsndfile (via soundfile), resampled to the model's
48 kHz rate with torchaudio's pure-tensor resampler, denoised, and streamed back
as a 48 kHz 16-bit WAV. wav/flac/ogg/mp3 decode in memory with no system
dependency; anything libsndfile cannot open (mp4/mov/mkv/webm video, m4a/aac)
is decoded by the host's ffmpeg, so video evidence is accepted as uploaded.
The upload's content decides, not its filename.

No authentication: this service is intended for trusted local-network use.
"""

import asyncio
import io
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from models.loader import enhance_waveform, get_sample_rate
from schemas.request import InfoResponse

router = APIRouter()

MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB


def _decode(raw: bytes):
    """Decode an upload to float32 ``(samples, channels)``.

    libsndfile first, from memory. On "Format not recognised" the upload goes
    to ffmpeg through a temp file, not a pipe: phone and camera mp4s keep their
    index (moov atom) at the end and cannot be read from a stream that does not
    seek. The ffmpeg path returns mono at the model rate, so no resample follows.

    Args:
        raw: Uploaded file bytes.

    Returns:
        The samples and their sample rate.

    Raises:
        ValueError: ffmpeg failed, e.g. the file has no audio track.
    """
    try:
        return sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except sf.SoundFileRuntimeError:
        pass
    sr = get_sample_rate()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(raw)
        tmp.flush()
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", tmp.name, "-vn", "-ac", "1",
             "-ar", str(sr), "-f", "f32le", "pipe:1"],
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode(errors="replace").replace(tmp.name, "upload")
            last = err.strip().splitlines()[-1:] or ["no audio samples"]
            raise ValueError(f"ffmpeg: {last[0]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 1), sr


def _run_enhance(data, orig_sr: int) -> bytes:
    """Blocking helper: resample -> enhance -> encode to WAV bytes.

    Runs in a thread-pool executor so it never blocks the event loop.

    Args:
        data: float32 samples shaped ``(samples, channels)``.
        orig_sr: Their sample rate.

    Returns:
        The enhanced audio as 48 kHz 16-bit WAV bytes.
    """
    import torch
    from torchaudio.functional import resample

    target_sr = get_sample_rate()

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
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)} bytes > {MAX_BYTES} byte limit).",
        )

    loop = asyncio.get_event_loop()
    try:
        data, orig_sr = await loop.run_in_executor(None, _decode, raw)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as e:
        raise HTTPException(
            status_code=415, detail=f"Unsupported or unreadable audio: {e}"
        )
    try:
        wav_bytes = await loop.run_in_executor(None, _run_enhance, data, orig_sr)
    except HTTPException:
        raise
    except Exception as e:  # inference failure
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