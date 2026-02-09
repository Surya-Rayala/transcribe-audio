

from __future__ import annotations

"""
Base types and interfaces for ASR backends.

This mirrors the pattern used in the `detect` package:

  - We define a canonical transcript schema (`asr-v1`).
  - Concrete backends (e.g. WhisperX) implement a thin interface that returns
    normalized results.
  - The orchestration layer (`audio_processing.asr`) owns:
        * run_dir / run_name creation
        * writing sidecar files (json, srt, vtt, txt, tsv)
        * optional subtitled video rendering

Backends should NOT do their own CLI parsing; they only expose a Python API.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union


SCHEMA_VERSION = "asr-v1"


# ---------------------------
# Canonical transcript types
# ---------------------------

class Word(TypedDict, total=False):
    start: float          # start time in seconds
    end: float            # end time in seconds
    text: str             # word text
    prob: Optional[float] # optional confidence
    speaker: Optional[str]


class Segment(TypedDict, total=False):
    start: float          # segment start (s)
    end: float            # segment end (s)
    text: str             # full segment text
    speaker: Optional[str]
    words: Optional[List[Word]]  # word-level breakdown (aligned)


class TranscriptResult(TypedDict, total=False):
    """
    Canonical result produced by a backend for a single transcribe() call.

    Orchestration code may enrich this with paths to sidecar files.
    """

    schema_version: str           # "asr-v1"
    backend: str                  # e.g. "whisperx"
    run_name: str                 # run identifier chosen by orchestration
    inputs: List[str]             # original inputs (audio/video paths as given)
    work_dir: str                 # directory used for intermediate/output files
    language: Optional[str]       # detected or specified language
    task: str                     # "transcribe" or "translate"
    diarization: bool             # whether diarization was applied
    alignment: bool               # whether alignment was applied
    segments: List[Segment]       # normalized segment list
    metadata: Dict[str, Any]      # backend-specific extras (raw outputs, versions, etc.)


# ---------------------------
# Abstract backend interface
# ---------------------------

class BaseASRBackend(ABC):
    """
    Abstract base class for ASR backends.

    Implementations should:
      - Initialize underlying models/resources in __init__ (or lazily).
      - Expose a .transcribe(...) method that:
            * Accepts normalized input paths (audio only; video is handled outside).
            * Respects key options (task, language, diarization, alignment, etc.).
            * Writes any backend-internal artifacts into work_dir as needed.
            * Returns a TranscriptResult using the canonical schema.

    The orchestration layer is responsible for:
      - Choosing run_name and creating run_dir.
      - Converting TranscriptResult into json/srt/vtt/tsv/txt files.
      - Handling subtitled video writing.
    """

    def __init__(self, **config: Any) -> None:
        # Store configuration for introspection / debugging.
        self.config: Dict[str, Any] = dict(config)
        # Default name is the lowercase class name, e.g. "whisperxbackend".
        # Orchestration code may override with a cleaner value.
        self.name: str = self.__class__.__name__.lower()

    @abstractmethod
    def transcribe(
        self,
        inputs: List[Path],
        *,
        task: str,
        language: Optional[str],
        work_dir: Path,
        print_progress: bool = False,
        **kwargs: Any,
    ) -> TranscriptResult:
        """
        Run ASR on one or more (audio) input files.

        Arguments:
          inputs:
            List of audio paths (video-to-audio extraction is done by the caller).
          task:
            "transcribe" (X->X) or "translate" (X->en), mirroring WhisperX.
          language:
            BCP-47 / whisper language code or None for auto-detect.
          work_dir:
            Directory for any backend-generated files (logits, alignments, etc.).
          print_progress:
            If True, backend may print progress logs.

        Returns:
          TranscriptResult:
            - schema_version must be "asr-v1"
            - backend should be a short identifier, e.g. "whisperx"
            - segments should be populated with normalized timings and text.
        """
        raise NotImplementedError


# ---------------------------
# Shared helpers
# ---------------------------

_ALLOWED_FORMATS = ("json", "txt", "srt", "vtt", "tsv", "all")


def normalize_output_formats(
    fmt: Union[str, List[str]],
    *,
    default: str = "json",
) -> List[str]:
    """
    Normalize CLI/API output_format input.

    Examples:
      "json"        -> ["json"]
      "all"         -> ["json", "txt", "srt", "vtt", "tsv"]
      ["srt", "vtt"]-> ["srt", "vtt"]
    """
    if isinstance(fmt, str):
        items = [t.strip().lower() for t in fmt.replace(";", ",").split(",") if t.strip()]
    else:
        items = [str(t).strip().lower() for t in fmt if str(t).strip()]

    if not items:
        items = [default]

    # Validate and expand "all"
    out: List[str] = []
    if "all" in items:
        out = [f for f in _ALLOWED_FORMATS if f != "all"]
    else:
        for t in items:
            if t not in _ALLOWED_FORMATS:
                raise ValueError(
                    f"Unsupported output_format '{t}'. Allowed: {', '.join(_ALLOWED_FORMATS)}"
                )
            if t != "all" and t not in out:
                out.append(t)
    return out


def default_run_name(
    inputs: List[Path],
    backend: str,
) -> str:
    """
    Construct a simple default run name based on the first input and backend.

    Example:
      inputs = ["call.mp4"], backend = "whisperx" -> "call_whisperx"
    """
    if not inputs:
        return f"asr_{backend}"
    first = Path(inputs[0])
    stem = first.stem or "input"
    return f"{stem}_{backend}"
    

__all__ = [
    "SCHEMA_VERSION",
    "Word",
    "Segment",
    "TranscriptResult",
    "BaseASRBackend",
    "normalize_output_formats",
    "default_run_name",
]