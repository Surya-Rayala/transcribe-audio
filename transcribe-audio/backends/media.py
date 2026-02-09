from __future__ import annotations

"""
Media utilities for audio_processing.backends.

Small helpers to:
  - detect whether an input is likely audio or video
  - extract audio from video using ffmpeg
  - write a subtitled video (burned-in subtitles) using ffmpeg

These are intentionally minimal and only used by the orchestration layer.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


# ---------------------------
# Generic helpers
# ---------------------------

def _has_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


# A small set of common extensions; not exhaustive, just practical.
_AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
}
_VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v",
}


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTS


def extract_audio(input_path: Path, work_dir: Path, *, overwrite: bool = True) -> Path:
    """
    Extract audio from a video file using ffmpeg and return the new audio path.

    Produces a .wav file in work_dir with the same stem as input_path.

    If input is already audio, returns it unchanged.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if is_audio(input_path):
        return input_path

    if not _has_ffmpeg():
        raise RuntimeError(
            "ffmpeg is required to extract audio from video, but was not found on PATH."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{input_path.stem}.wav"

    if out_path.exists() and not overwrite:
        return out_path

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",               # no video
        "-acodec",
        "pcm_s16le",         # standard 16-bit PCM
        "-ar",
        "16000",             # 16 kHz (reasonable default for ASR)
        "-ac",
        "1",                 # mono
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to extract audio from {input_path}") from e

    if not out_path.exists():
        raise RuntimeError(f"Expected audio file not created: {out_path}")

    return out_path


def write_subtitled_video(
    src_media: PathLike,
    subtitles_path: PathLike,
    output_path: PathLike,
    *,
    overwrite: bool = True,
    fps: int = 30,
    video_resolution: str = "1280x720",
    audio_strip_resolution: str = "1280x200",
    bg_color: str = "black",
) -> Optional[Path]:
    """
    Create a subtitled video from a source media file and a subtitle file using ffmpeg.

    Behaviors:
      - If src_media is a video file: burn subtitles directly onto the video.
      - If src_media is an audio file: create a simple video with a solid-color background,
        burn subtitles onto it, and use the audio as the soundtrack.

    For audio-only inputs, a short letterbox-style black strip video is created (default 1280x200) so subtitles are visible without a huge empty frame.
    For audio-only renders, subtitles are drawn with a larger font so they remain readable at the reduced height.

    Returns:
      - Path to the created subtitled video on success.
      - None if ffmpeg is not available or if the command fails (fails softly for personal use).
    """
    src = Path(src_media)
    subs = Path(subtitles_path)
    out = Path(output_path)

    if not src.exists():
        raise FileNotFoundError(f"Source media not found: {src}")
    if not subs.exists():
        raise FileNotFoundError(f"Subtitles file not found: {subs}")

    if not _has_ffmpeg():
        print("[warn] ffmpeg not found; skipping subtitled video generation.")
        return None

    # Ensure mp4 extension and parent directory
    if out.suffix.lower() != ".mp4":
        out = out.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    ow_flag = "-y" if overwrite else "-n"
    subs_arg = subs.as_posix()

    if is_video(src):
        # Case 1: real video input -> burn subtitles on top of existing video
        cmd = [
            "ffmpeg",
            ow_flag,
            "-i",
            str(src),
            "-vf",
            f"subtitles={subs_arg}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(out),
        ]
    else:
        # Case 2: audio-only input -> generate a black background video with subtitles + audio
        cmd = [
            "ffmpeg",
            ow_flag,
            # Video: solid color background
            "-f",
            "lavfi",
            "-i",
            f"color=c={bg_color}:s={audio_strip_resolution}:r={fps}",
            # Audio:
            "-i",
            str(src),
            # Burn subtitles onto the generated video
            "-vf",
            f"subtitles={subs_arg}:force_style='Fontsize=60'",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print("[warn] ffmpeg failed to create subtitled video; skipping.")
        return None

    if not out.exists():
        print("[warn] Subtitled video was not created as expected; skipping.")
        return None

    return out.resolve()


__all__ = [
    "is_audio",
    "is_video",
    "extract_audio",
    "write_subtitled_video",
]