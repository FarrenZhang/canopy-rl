# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Loads the Canopy vLLM hooks at interpreter start when this directory is on PYTHONPATH."""
import os

if os.environ.get("CANOPY_ENGINE_HOOKS", "0") == "1":
    import vllm_hooks

    vllm_hooks.install()
