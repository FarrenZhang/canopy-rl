# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""In-process stand-ins for Ray actor handles, so coordinator logic runs without Ray."""

from __future__ import annotations

import inspect
from typing import Any


class _Method:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any):
        async def _call():
            out = self._fn(*args, **kwargs)
            if inspect.isawaitable(out):
                out = await out
            return out

        return _call()


class LocalHandle:
    """``handle.method.remote(...)`` returns an awaitable, as with a Ray actor handle.

    A missing method raises ``AttributeError`` on access, as Ray handles do.
    """

    def __init__(self, obj: Any):
        self._obj = obj

    def __getattr__(self, name: str) -> _Method:
        if name.startswith("__"):
            raise AttributeError(name)
        return _Method(getattr(self._obj, name))
