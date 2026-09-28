"""Gate tests: POST /api/v1/enhance accepts video and m4a uploads.

The app team's UC sheet (AmThanh_AI.xlsx) feeds this service audio and video
evidence. Before the fix the route answered 415 to any filename not ending in
.wav/.flac/.ogg/.mp3, and decoding was libsndfile only, so an mp4 could not
pass even when renamed. No model is loaded: enhance_waveform is the identity
and the sample rate is pinned to DeepFilterNet3's 48 kHz. Fixtures are made
with the host's ffmpeg, the binary the route calls.

Run:
    enhance_env/bin/python -m unittest discover -s tests -v
"""

import asyncio
import functools
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import HTTPException  # noqa: E402
from starlette.datastructures import UploadFile  # noqa: E402

from routes import enhance as route  # noqa: E402

SINE = ["-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration={d}"]
VIDEO = ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d={d}"]


def make(inputs: list[str], codecs: list[str], suffix: str, seconds: float) -> bytes:
    """Render a lavfi fixture with ffmpeg and return the file bytes.

    Args:
        inputs: ffmpeg input arguments, with ``{d}`` standing for the duration.
        codecs: ffmpeg output codec arguments.
        suffix: Output file suffix, which picks the container.
        seconds: Fixture duration.

    Returns:
        The encoded file contents.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "fixture" + suffix)
        args = [a.format(d=seconds) for a in inputs]
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", *args, *codecs, "-shortest", out],
            check=True,
        )
        with open(out, "rb") as fh:
            return fh.read()


@functools.cache
def fixtures() -> dict[str, bytes]:
    """Render every fixture once per test run.

    Returns:
        Fixture name to file bytes.
    """
    return {
        "mp4": make(SINE + VIDEO, ["-c:v", "mpeg4", "-c:a", "aac"], ".mp4", 3.0),
        "m4a": make(SINE, ["-c:a", "aac"], ".m4a", 3.0),
        "silent_mp4": make(VIDEO, ["-c:v", "mpeg4"], ".mp4", 3.0),
        "stereo_wav": make(SINE, ["-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le"], ".wav", 3.0),
    }


def post(name: str, data: bytes) -> tuple[int, bytes]:
    """Call the route handler the way FastAPI does and collect the body.

    Args:
        name: Upload filename.
        data: Upload contents.

    Returns:
        HTTP status and response body.
    """

    async def run():
        """Await the handler and drain its streaming body.

        Returns:
            HTTP status and response body.
        """
        resp = await route.enhance_audio(
            UploadFile(file=io.BytesIO(data), filename=name)
        )
        return resp.status_code, b"".join([c async for c in resp.body_iterator])

    return asyncio.run(run())


class VideoUploadTests(unittest.TestCase):
    """The upload's content, not its filename, decides whether it decodes."""

    def setUp(self):
        for patcher in (
            mock.patch.object(route, "enhance_waveform", side_effect=lambda a: a),
            mock.patch.object(route, "get_sample_rate", return_value=48000),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_wav(self, body: bytes, channels: int) -> None:
        """Check the response is the promised 48 kHz 16-bit WAV of the 3 s fixture.

        Args:
            body: Response body.
            channels: Expected channel count.
        """
        info = sf.info(io.BytesIO(body))
        self.assertEqual((info.format, info.subtype), ("WAV", "PCM_16"))
        self.assertEqual(info.samplerate, 48000)
        self.assertEqual(info.channels, channels)
        self.assertAlmostEqual(info.duration, 3.0, delta=0.1)

    def test_mp4_video_is_denoised(self):
        status, body = post("clip.mp4", fixtures()["mp4"])
        self.assertEqual(status, 200)
        self.assert_wav(body, channels=1)

    def test_m4a_is_denoised(self):
        status, body = post("voice.m4a", fixtures()["m4a"])
        self.assertEqual(status, 200)
        self.assert_wav(body, channels=1)

    def test_filename_does_not_decide(self):
        status, body = post("upload.bin", fixtures()["mp4"])
        self.assertEqual(status, 200)
        with self.assertRaises(HTTPException) as ctx:
            post("clip.wav", b"not audio at all" * 100)
        self.assertEqual(ctx.exception.status_code, 415)

    def test_video_without_audio_track_is_415(self):
        with self.assertRaises(HTTPException) as ctx:
            post("clip.mp4", fixtures()["silent_mp4"])
        self.assertEqual(ctx.exception.status_code, 415)
        self.assertIn("Unsupported or unreadable audio", ctx.exception.detail)

    def test_native_wav_keeps_channels_and_skips_ffmpeg(self):
        with mock.patch.object(route.subprocess, "run") as run:
            status, body = post("stereo.wav", fixtures()["stereo_wav"])
        run.assert_not_called()
        self.assertEqual(status, 200)
        self.assert_wav(body, channels=2)

    def test_empty_upload_is_400(self):
        with self.assertRaises(HTTPException) as ctx:
            post("clip.mp4", b"")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_temp_file_is_removed(self):
        seen = []
        real = tempfile.NamedTemporaryFile

        def spy(*args, **kwargs):
            """Record the path of each temp file the decoder creates.

            Args:
                *args: Passed through.
                **kwargs: Passed through.

            Returns:
                The real temp file.
            """
            fh = real(*args, **kwargs)
            seen.append(fh.name)
            return fh

        with mock.patch.object(route.tempfile, "NamedTemporaryFile", spy):
            post("clip.mp4", fixtures()["mp4"])
            with self.assertRaises(HTTPException):
                post("clip.mp4", fixtures()["silent_mp4"])
        self.assertEqual(len(seen), 2)
        self.assertFalse(any(os.path.exists(p) for p in seen))


if __name__ == "__main__":
    unittest.main()
