"""Gate test: the model loads on a box without git (c09 may have none).

DeepFilterNet's logger asks git for a commit hash at init_df and only catches
CalledProcessError, so with no git binary init_df raises FileNotFoundError and
the service never starts. Found by ai-audio-jobs evals/e2e_c09_bundle.sh in an
Ubuntu container without git.

Run: enhance_env/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import df.logger  # noqa: E402

from models import loader  # noqa: E402


class NoGit(unittest.TestCase):
    """init_df's commit-hash lookup must survive a missing git binary."""

    def test_commit_hash_without_git(self):
        """With git not on PATH, the lookup the logger uses returns None."""
        with mock.patch.dict(os.environ, {"PATH": "/nonexistent"}):
            loader.tolerate_missing_git()
            self.assertIsNone(df.logger.get_commit_hash())


if __name__ == "__main__":
    unittest.main()
