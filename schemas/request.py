"""Pydantic models for the DeepFilterNet API.

The primary endpoint is file-based (audio in, audio out), so most requests use
multipart/form-data rather than JSON. These models document the JSON responses
and the metadata endpoint.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model: str = Field(..., examples=["DeepFilterNet3"])
    device: str = Field(..., examples=["cpu"])
    sample_rate: int = Field(..., examples=[48000])


class InfoResponse(BaseModel):
    model: str = Field(..., description="Loaded DeepFilterNet model name")
    device: str = Field(..., description="cpu or cuda")
    sample_rate: int = Field(..., description="Model native sample rate in Hz")
    post_filter: bool = Field(..., description="Whether the post-filter is enabled")


class ErrorResponse(BaseModel):
    error: str
