"""Model loading and inference for DeepFilterNet speech enhancement.

DeepFilterNet is an audio-in / audio-out model: it takes a noisy waveform and
returns an enhanced (denoised) waveform at 48 kHz. We use the official
`deepfilternet` PyPI package, which bundles the pretrained weights, so there is
no separate checkpoint file to download -- ``init_df()`` loads them (downloading
to a local cache on first run).

The model is loaded once at startup (see main.py lifespan) and reused for every
request. For multiple uvicorn workers, each process loads its own copy.

Device note: DeepFilterNet's own ``enhance()`` selects the compute device via
``df.modules.get_device()`` (CUDA if visible, else CPU) and places feature
tensors there. We therefore do NOT force a different device on the model -- that
would cause a model/feature device mismatch. To force CPU on a CUDA box, hide
the GPU with ``CUDA_VISIBLE_DEVICES=""`` before launch.
"""

import os
import threading

import numpy as np
import torch

# Model name: one of "DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3".
# DeepFilterNet3 is the newest and the recommended default.
MODEL_NAME = os.getenv("MODEL_NAME", "DeepFilterNet3")
# Post-filter slightly over-attenuates very noisy sections. Off by default.
POST_FILTER = os.getenv("POST_FILTER", "false").lower() in ("1", "true", "yes")

_model = None
_df_state = None
_lock = threading.Lock()


def tolerate_missing_git() -> None:
    """Let init_df run on a box without git (c09 may have none).

    DeepFilterNet's logger asks git for a commit hash and catches only
    CalledProcessError, so a missing git binary (FileNotFoundError) stops the
    model from loading. The hash only goes into a log line.
    """
    import df.logger
    import df.utils

    def commit_hash():
        """Return df's git commit hash, or None without git.

        Returns:
            The hash string, or None.
        """
        try:
            return df.utils.get_commit_hash()
        except FileNotFoundError:
            return None

    df.logger.get_commit_hash = commit_hash


def get_device() -> str:
    """Report the device DeepFilterNet will actually use."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    """Load the DeepFilterNet model + DF state once and cache it.

    Returns a tuple ``(model, df_state)``. ``df_state`` carries the STFT
    configuration and exposes ``df_state.sr()`` -- the model's native sample
    rate (48000 Hz). ``init_df`` already moves the model to the correct device.
    """
    global _model, _df_state
    if _model is None:
        with _lock:
            if _model is None:  # double-checked locking
                from df.enhance import init_df  # lazy import

                tolerate_missing_git()

                # init_df returns (model, df_state, suffix[, epoch]) -- the
                # released 0.5.x gives 3 values, GitHub main gives 4. Take the
                # first two so both versions work.
                result = init_df(
                    model_base_dir=MODEL_NAME,
                    post_filter=POST_FILTER,
                )
                model, df_state = result[0], result[1]
                model.eval()
                _model, _df_state = model, df_state
    return _model, _df_state


def get_sample_rate() -> int:
    """Return the model's native sample rate (Hz)."""
    _, df_state = get_model()
    return df_state.sr()


def enhance_waveform(audio: np.ndarray) -> np.ndarray:
    """Denoise a waveform that is already at the model sample rate (48 kHz).

    Args:
        audio: float32 array shaped ``(channels, samples)``.

    Returns:
        Enhanced waveform as a float32 ``(channels, samples)`` array.
    """
    from df.enhance import enhance  # lazy import; handles no_grad + device

    model, df_state = get_model()
    tensor = torch.from_numpy(np.ascontiguousarray(audio))
    enhanced = enhance(model, df_state, tensor)
    return enhanced.cpu().numpy()
