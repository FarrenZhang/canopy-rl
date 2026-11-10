# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Engine-side hooks for vLLM V1 (see :mod:`canopy.engine.vllm_hooks`).

``HOOKS_DIR`` is the directory to put on a vLLM server's ``PYTHONPATH``,
together with ``CANOPY_ENGINE_HOOKS=1``, so that ``sitecustomize.py`` loads the
hooks in every process of the server.
"""

import os

HOOKS_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ["HOOKS_DIR"]
