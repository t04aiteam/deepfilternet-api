"""FastAPI entrypoint for the DeepFilterNet speech-enhancement API.

Wraps the DeepFilterNet real-time noise-suppression model
(https://github.com/Rikorose/DeepFilterNet) behind a small REST API.
The model is warmed up once at startup so the first request isn't slow.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.loader import MODEL_NAME, get_device, get_model, get_sample_rate
from routes.enhance import router
from schemas.request import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up at startup: fails fast if weights can't be loaded.
    get_model()
    yield


app = FastAPI(
    title="DeepFilterNet Speech Enhancement API",
    description="Upload noisy audio, get a denoised 48 kHz WAV back.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness/readiness probe. Confirms the model is loaded."""
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        device=get_device(),
        sample_rate=get_sample_rate(),
    )


@app.exception_handler(Exception)
async def global_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7551)