# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared KDA test fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def kimi_k3_checkpoint_dir() -> Path:
    """Return the explicitly selected pinned Kimi-K3 checkpoint subset."""
    value = os.getenv("KIMI_K3_CKPT")
    if value is None:
        pytest.skip("set KIMI_K3_CKPT to the pinned Kimi-K3 checkpoint subset")
    return Path(value)
