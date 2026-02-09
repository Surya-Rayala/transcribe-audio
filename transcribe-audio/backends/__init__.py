

from __future__ import annotations

"""
Backend registry and tiny factory for audio_processing.

Usage:

    from audio_processing.backends import create_backend
    backend = create_backend(
        name="whisperx",
        model="small",
        device="cuda",
    )

This mirrors the style of `detect.detectors.__init__`:
  - explicit backend map
  - thin helpers for discovery and construction
"""

from typing import Any, Dict, List

from .base import BaseASRBackend
from .whisperx_backend import WhisperXBackend


_BACKENDS: Dict[str, type[BaseASRBackend]] = {
    "whisperx": WhisperXBackend,
}


def available_backends() -> List[str]:
    """Return a sorted list of available backend names."""
    return sorted(_BACKENDS.keys())


def create_backend(name: str, **kwargs: Any) -> BaseASRBackend:
    """Instantiate a backend by name."""
    key = (name or "").strip().lower()
    if key not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {', '.join(available_backends())}"
        )
    cls = _BACKENDS[key]
    return cls(**kwargs)


__all__ = [
    "BaseASRBackend",
    "available_backends",
    "create_backend",
]