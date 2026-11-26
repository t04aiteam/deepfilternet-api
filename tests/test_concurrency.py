"""Overlapping requests reach DeepFilterNet one at a time."""

import asyncio
import os
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib  # noqa: E402

from models import loader  # noqa: E402


class ConcurrentEnhanceTest(unittest.IsolatedAsyncioTestCase):
    async def test_overlapping_requests_run_one_enhance_at_a_time(self):
        """The model is shared; the deleted CUDA lock's job now belongs to IdleModel."""
        state = {"running": 0, "most": 0}
        lock = threading.Lock()

        def fake_enhance(model, df_state, tensor):
            """Hold the call open briefly so overlapping calls would show.

            Args:
                model: Ignored.
                df_state: Ignored.
                tensor: The input audio.

            Returns:
                The input unchanged.
            """
            with lock:
                state["running"] += 1
                state["most"] = max(state["most"], state["running"])
            time.sleep(0.05)
            with lock:
                state["running"] -= 1
            return tensor

        audio = np.zeros((1, 4800), dtype=np.float32)
        with (
            mock.patch.object(loader, "_model", object()),
            mock.patch.object(loader, "_df_state", object()),
            mock.patch.object(importlib.import_module("df.enhance"), "enhance", fake_enhance),
        ):
            async with loader.idle_model.lifespan(preload=True):
                out = await asyncio.gather(
                    *(
                        loader.idle_model.run(loader.enhance_waveform, audio)
                        for _ in range(4)
                    )
                )
        self.assertEqual(len(out), 4)
        self.assertEqual(state["most"], 1)


if __name__ == "__main__":
    unittest.main()
