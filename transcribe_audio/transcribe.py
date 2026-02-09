from __future__ import annotations

"""
ASR orchestration: CLI + importable function.

This mirrors the `detect.detect_video` pattern:

  - Accept audio or video inputs.
  - Normalize into a run directory with a run name.
  - Delegate model-specific work to a backend (e.g. WhisperX).
  - Export a canonical JSON transcript plus optional txt/srt/vtt/tsv.
  - Optionally create a subtitled video with burned-in subtitles.

Usage (CLI):

    python -m audio_processing.asr \
        --input path/to/file.mp4 \
        --backend whisperx \
        --model small \
        --task transcribe \
        --language en \
        --output-dir out \
        --run-name demo_run \
        --output-format all \
        --save-subtitled-video demo_subtitled.mp4

Usage (import):

    from audio_processing.asr import transcribe

    result = transcribe(
        inputs="in.mp4",
        backend="whisperx",
        model="small",
        output_format=["json", "srt"],
        run_name="demo_run",
        save_subtitled_video="demo_subtitled.mp4",
    )
    print(result["run_dir"])
    print(result["outputs"]["json"])
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Union

# Flexible imports to support:
#   - python -m audio_processing.asr
#   - python audio_processing/asr.py
if __package__ in (None, ""):
    import sys

    PKG_DIR = Path(__file__).resolve().parent
    sys.path.append(str(PKG_DIR.parent))  # project root
    try:
        from audio_processing.backends import (  # type: ignore
            available_backends,
            create_backend,
        )
        from audio_processing.backends.base import (  # type: ignore
            normalize_output_formats,
            default_run_name,
            TranscriptResult,
        )
        from audio_processing.backends.media import (  # type: ignore
            is_video,
            extract_audio,
            write_subtitled_video,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Failed to import audio_processing backends. "
            "Run this module from the project root or install the package."
        ) from e
else:
    from .backends import available_backends, create_backend
    from .backends.base import normalize_output_formats, default_run_name, TranscriptResult
    from .backends.media import is_video, extract_audio, write_subtitled_video


PathLike = Union[str, Path]


# ---------------------------
# Public API
# ---------------------------

def transcribe(
    inputs: Union[PathLike, Iterable[PathLike]],
    *,
    backend: str = "whisperx",
    output_dir: PathLike = "out",
    run_name: Optional[str] = None,
    output_format: Union[str, List[str]] = "json",
    save_subtitled_video: Optional[PathLike] = None,
    task: str = "transcribe",
    language: Optional[str] = None,
    print_progress: bool = False,
    **backend_options: Any,
) -> Dict[str, Any]:
    """
    Run ASR on one or more audio/video inputs.

    Arguments:
      inputs:
        Single path or iterable of paths (audio or video).
      backend:
        Backend name (default: "whisperx").
      output_dir:
        Root directory for runs (default: "./out").
      run_name:
        Optional run identifier; if omitted, derived from first input + backend.
      output_format:
        One of "json", "txt", "srt", "vtt", "tsv", "all" or a list thereof.
      save_subtitled_video:
        Optional filename (relative or absolute). If provided and at least one
        video input exists, a subtitled video is created inside run_dir using
        the generated SRT track (best-effort).
      task, language, print_progress, backend_options:
        Passed through to the backend, allowing you to control all relevant
        WhisperX-style parameters (model, device, VAD, diarization, etc.).

    Returns:
      {
        "run_dir": str,
        "outputs": {
          "json": "path/to/transcript.json",
          "txt": "path/to/transcript.txt",
          "srt": "path/to/transcript.srt",
          ...
        },
        "subtitled_video": "path/to/subtitled.mp4"  # if created
      }
    """
    # Normalize input list
    if isinstance(inputs, (str, Path)):
        input_paths = [Path(inputs)]
    else:
        input_paths = [Path(p) for p in inputs]

    if not input_paths:
        raise ValueError("No inputs provided to transcribe().")

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    # Backend sanity
    available = available_backends()
    if backend not in available:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {', '.join(available)}"
        )

    # Run directory
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if run_name is None or not str(run_name).strip():
        run_name = default_run_name(input_paths, backend)
    run_name = str(run_name)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional progress bar for preprocessing
    try:
        from tqdm import tqdm  # type: ignore
        _prep_iter = tqdm(input_paths, desc="Preparing audio", unit="file", disable=not print_progress)
    except Exception:  # pragma: no cover
        _prep_iter = input_paths
    # Prepare audio inputs (extract from videos where needed)
    audio_dir = run_dir / "audio"
    audio_inputs: List[Path] = []
    first_video_input: Optional[Path] = None

    for src in _prep_iter:
        if is_video(src):
            if first_video_input is None:
                first_video_input = src
            audio_path = extract_audio(src, audio_dir, overwrite=True)
            audio_inputs.append(audio_path)
        else:
            audio_inputs.append(src)

    # Normalize output formats
    formats = normalize_output_formats(output_format)

    # Instantiate backend with configuration (run_name included for metadata)
    backend_instance = create_backend(
        name=backend,
        run_name=run_name,
        **backend_options,
    )

    # Call backend
    transcript: TranscriptResult = backend_instance.transcribe(
        inputs=audio_inputs,
        task=task,
        language=language,
        work_dir=run_dir,
        print_progress=print_progress,
        **backend_options,
    )

    # Ensure run_name and inputs are populated (backend is allowed to ignore)
    transcript.setdefault("run_name", run_name)
    transcript.setdefault("inputs", [str(p) for p in input_paths])
    transcript.setdefault("work_dir", str(run_dir))

    # Write canonical JSON transcript
    outputs: Dict[str, str] = {}
    json_path = run_dir / "transcript.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    outputs["json"] = str(json_path)

    segs = transcript.get("segments", []) or []

    # Sidecar writers
    if "txt" in formats:
        txt_path = run_dir / "transcript.txt"
        _write_txt(segs, txt_path)
        outputs["txt"] = str(txt_path)

    srt_path: Optional[Path] = None
    if "srt" in formats:
        srt_path = run_dir / "transcript.srt"
        _write_srt(segs, srt_path)
        outputs["srt"] = str(srt_path)

    if "vtt" in formats:
        vtt_path = run_dir / "transcript.vtt"
        _write_vtt(segs, vtt_path)
        outputs["vtt"] = str(vtt_path)

    if "tsv" in formats:
        tsv_path = run_dir / "transcript.tsv"
        _write_tsv(segs, tsv_path)
        outputs["tsv"] = str(tsv_path)

    # Optional subtitled video
    subtitled_video_path: Optional[Path] = None
    if save_subtitled_video is not None:
        # Require an SRT track; if user didn't request one explicitly, create it.
        if srt_path is None:
            srt_path = run_dir / "transcript.srt"
            _write_srt(segs, srt_path)
            outputs.setdefault("srt", str(srt_path))

        # Choose source:
        # - Prefer the first real video input, if any.
        # - Otherwise fall back to the first audio input, and media.write_subtitled_video
        #   will generate a compact black-strip video with subtitles.
        if first_video_input is not None:
            src_for_video = first_video_input
        else:
            # Assumes at least one audio input was provided earlier.
            src_for_video = audio_inputs[0]

        # Resolve target path
        if save_subtitled_video == "__AUTO__":
            base_name = f"{Path(src_for_video).stem}_subtitled.mp4"
            target = run_dir / base_name
        else:
            target = Path(save_subtitled_video)
            if not target.is_absolute():
                target = run_dir / target.name

        subtitled_video_path = write_subtitled_video(
            src_for_video,
            srt_path,
            target,
            overwrite=True,
        )

    result: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "outputs": outputs,
    }
    if subtitled_video_path is not None:
        result["subtitled_video"] = str(subtitled_video_path)

    return result


# ---------------------------
# Formatting helpers
# ---------------------------

def _format_ts_srt(t: float) -> str:
    """Format seconds as SRT timestamp."""
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000.0))
    h = ms // (3600_000)
    ms -= h * 3600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"



def _format_ts_vtt(t: float) -> str:
    """Format seconds as WebVTT timestamp."""
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000.0))
    h = ms // (3600_000)
    ms -= h * 3600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# --- Helper functions for segment timing ---

def _is_finite_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def _derive_start_end(seg: Mapping[str, Any]) -> tuple[float, float]:
    """Derive start/end for a segment.

    Prefer word-level timestamps (min word.start, max word.end) when present.
    Fall back to segment-level start/end otherwise.
    """
    seg_start = float(seg.get("start", 0.0) or 0.0)
    seg_end = float(seg.get("end", seg_start) or seg_start)

    words = seg.get("words")
    if isinstance(words, list) and words:
        w_starts: List[float] = []
        w_ends: List[float] = []
        for w in words:
            if not isinstance(w, Mapping):
                continue
            ws = w.get("start")
            we = w.get("end")
            if _is_finite_number(ws):
                w_starts.append(float(ws))
            if _is_finite_number(we):
                w_ends.append(float(we))
        if w_starts and w_ends:
            seg_start = min(w_starts)
            seg_end = max(w_ends)

    if seg_end <= seg_start:
        seg_end = seg_start + 0.01
    return seg_start, seg_end


def _write_txt(segments: List[Mapping[str, Any]], path: Path) -> None:
    lines: List[str] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_srt(segments: List[Mapping[str, Any]], path: Path) -> None:
    lines: List[str] = []
    idx = 1
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start, end = _derive_start_end(seg)
        speaker = seg.get("speaker")
        lines.append(str(idx))
        lines.append(f"{_format_ts_srt(start)} --> {_format_ts_srt(end)}")
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
        lines.append("")
        idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_vtt(segments: List[Mapping[str, Any]], path: Path) -> None:
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start, end = _derive_start_end(seg)
        speaker = seg.get("speaker")
        lines.append(f"{_format_ts_vtt(start)} --> {_format_ts_vtt(end)}")
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tsv(segments: List[Mapping[str, Any]], path: Path) -> None:
    lines: List[str] = ["start\tend\tspeaker\ttext"]
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start, end = _derive_start_end(seg)
        speaker = seg.get("speaker") or ""
        # Escape tabs/newlines in text
        safe_text = text.replace("\t", " ").replace("\n", " ")
        lines.append(f"{start:.3f}\t{end:.3f}\t{speaker}\t{safe_text}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# CLI
# ---------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run ASR on audio/video files and export transcripts.\n\n"
            "Typical usage:\n"
            "  python -m audio_processing.asr -i input.mp4 -o out -f all\n\n"
            "Notes:\n"
            "  • Video inputs are automatically converted to 16kHz mono WAV before ASR.\n"
            "  • Outputs are written under: <output-dir>/<run-name>/\n"
            "  • Use --diarize for speaker labels (requires HF token for pyannote models).\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------
    # Core inputs / outputs
    # ---------------------------
    g_io = ap.add_argument_group("Inputs & outputs")

    g_io.add_argument(
        "--input",
        "-i",
        dest="inputs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Audio/video file(s) to transcribe. "
            "Video files will be audio-extracted via ffmpeg into the run directory."
        ),
    )

    g_io.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("out"),
        help=(
            "Root directory for outputs. A per-run folder is created inside this directory."
        ),
    )

    g_io.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Name of the run folder under --output-dir. "
            "If omitted, it is derived from the first input filename and backend "
            "(e.g., 'meeting_whisperx')."
        ),
    )

    g_io.add_argument(
        "--output-format",
        "-f",
        dest="output_format",
        type=str,
        default="json",
        choices=["json", "txt", "srt", "vtt", "tsv", "all"],
        help=(
            "Which transcript files to generate. "
            "json = canonical output; "
            "srt/vtt = subtitles; "
            "tsv = tab-separated segments; "
            "all = json+txt+srt+vtt+tsv."
        ),
    )

    g_io.add_argument(
        "--save-subtitled-video",
        nargs="?",
        const="__AUTO__",
        type=str,
        default=None,
        help=(
            "Create an .mp4 with burned-in subtitles (uses the generated SRT track).\n"
            "• If a video input exists: subtitles are burned onto the FIRST video.\n"
            "• If inputs are audio-only: creates a small black-strip video with subtitles.\n"
            "Usage:\n"
            "  --save-subtitled-video           (auto-name inside run folder)\n"
            "  --save-subtitled-video out.mp4   (explicit filename)\n"
        ),
    )

    # ---------------------------
    # Backend selection
    # ---------------------------
    g_backend = ap.add_argument_group("Backend")
    g_backend.add_argument(
        "--backend",
        type=str,
        default="whisperx",
        choices=available_backends(),
        help=(
            "ASR backend to use. This package currently exposes registered backends "
            "via audio_processing.backends.available_backends()."
        ),
    )

    # ---------------------------
    # Model / compute settings
    # ---------------------------
    g_model = ap.add_argument_group("Model & performance")

    g_model.add_argument(
        "--model",
        type=str,
        default="small",
        help=(
            "ASR model size/name. Larger models are generally more accurate but slower "
            "and use more memory. "
            "For WhisperX/faster-whisper models see: "
            "https://huggingface.co/collections/Systran/faster-whisper"
        ),
    )
    g_model.add_argument(
        "--model_cache_only",
        type=_str2bool,
        default=False,
        help=(
            "If true, do not download models; only use locally cached files. "
            "Useful in offline environments (will fail if the model isn't cached)."
        ),
    )
    g_model.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to use for model downloads/cache (download_root). "
            "Leave unset to use the default cache location."
        ),
    )
    g_model.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Compute device. Examples: 'cuda', 'cpu'. "
            "If omitted, backend auto-selects (typically CUDA if available)."
        ),
    )
    g_model.add_argument(
        "--device_index",
        type=int,
        default=0,
        help="GPU device index (when using CUDA).",
    )
    g_model.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "Batch size for ASR decoding. Higher can be faster but uses more VRAM. "
            "If you hit out-of-memory on GPU, reduce this."
        ),
    )
    g_model.add_argument(
        "--compute_type",
        type=str,
        default="float32",
        choices=["float16", "float32", "int8"],
        help=(
            "Precision / quantization. "
            "float16 is faster on modern GPUs but may be less stable; "
            "float32 is safest; "
            "int8 reduces memory and can be faster on CPU, sometimes with quality impact."
        ),
    )

    g_task = ap.add_argument_group("Task")
    g_task.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help=(
            "transcribe = speech-to-text in the same language; "
            "translate = translate speech to English (if supported by the model)."
        ),
    )
    g_task.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Language code (e.g. 'en', 'es'). If omitted, auto-detect is used. "
            "Setting this can improve accuracy/speed when you know the language."
        ),
    )

    # ---------------------------
    # Alignment (word-level timestamps)
    # ---------------------------
    g_align = ap.add_argument_group("Alignment (word-level timestamps)")

    g_align.add_argument(
        "--no_align",
        action="store_true",
        help=(
            "Disable alignment. If set, you may get less precise timestamps and no "
            "word-level timing info. Turning alignment off can be faster."
        ),
    )
    g_align.add_argument(
        "--align_model",
        type=str,
        default=None,
        help=(
            "Optional alignment model name. If unset, WhisperX chooses a default per language. "
            "See: https://docs.pytorch.org/audio/0.12.0/pipelines.html#wav2vec-2-0-hubert-fine-tuned-asr"
        ),
    )
    g_align.add_argument(
        "--interpolate_method",
        type=str,
        default="nearest",
        choices=["nearest", "linear", "ignore"],
        help=(
            "How to handle missing alignment timestamps. "
            "nearest = fill with nearest neighbors; "
            "linear = interpolate; "
            "ignore = keep missing values (can create gaps)."
        ),
    )
    g_align.add_argument(
        "--return_char_alignments",
        action="store_true",
        help="If set, also return character-level alignments when supported (more detailed, larger output).",
    )

    # ---------------------------
    # VAD (speech detection / chunking)
    # ---------------------------
    g_vad = ap.add_argument_group("VAD (speech detection & chunking)")

    g_vad.add_argument(
        "--vad_method",
        type=str,
        default="pyannote",
        choices=["pyannote", "silero"],
        help=(
            "Voice activity detection backend. Affects how audio is chunked into speech regions. "
            "Different methods can change segmentation quality."
        ),
    )
    g_vad.add_argument(
        "--vad_onset",
        type=float,
        default=0.500,
        help=(
            "Speech start threshold for VAD. Higher = fewer false positives but may miss quiet speech."
        ),
    )
    g_vad.add_argument(
        "--vad_offset",
        type=float,
        default=0.363,
        help=(
            "Speech end threshold for VAD. Higher = cuts off sooner; lower = keeps trailing audio longer."
        ),
    )
    g_vad.add_argument(
        "--chunk_size",
        type=int,
        default=30,
        help=(
            "Chunk size in seconds used during transcription. "
            "Smaller chunks can reduce memory and sometimes improve robustness; "
            "larger chunks can be faster but risk quality drops on long audio."
        ),
    )

    # ---------------------------
    # Diarization (speaker labels)
    # ---------------------------
    g_diar = ap.add_argument_group("Diarization (speaker labels)")

    g_diar.add_argument(
        "--diarize",
        action="store_true",
        help=(
            "Enable diarization (assign speaker labels like SPEAKER_00). "
            "Usually requires a Hugging Face token if using pyannote models."
        ),
    )
    g_diar.add_argument(
        "--min_speakers",
        type=int,
        default=None,
        help="Optional lower bound on number of speakers (helps diarization when you have a good guess).",
    )
    g_diar.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="Optional upper bound on number of speakers (helps diarization when you have a good guess).",
    )
    g_diar.add_argument(
        "--diarize_model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Diarization model id (pyannote pipeline). See: https://huggingface.co/pyannote/models",
    )
    g_diar.add_argument(
        "--speaker_embeddings",
        action="store_true",
        help=(
            "If set, compute and use speaker embeddings during diarization. "
            "Can improve speaker consistency but increases runtime/memory."
        ),
    )

    # ---------------------------
    # Decoding / sampling knobs (advanced)
    # ---------------------------
    g_dec = ap.add_argument_group("Decoding & advanced ASR knobs")

    g_dec.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. 0.0 is most deterministic; higher can help tricky audio but may add errors.")
    g_dec.add_argument("--best_of", type=int, default=5, help="Number of candidates sampled (used with non-zero temperature). Higher = slower, sometimes better.")
    g_dec.add_argument("--beam_size", type=int, default=5, help="Beam search width. Higher = slower, sometimes more accurate.")
    g_dec.add_argument("--patience", type=float, default=1.0, help="Beam search patience. Higher explores longer, can improve accuracy at cost of speed.")
    g_dec.add_argument("--length_penalty", type=float, default=1.0, help="Penalize/encourage longer outputs in decoding.")

    g_dec.add_argument("--suppress_tokens", type=str, default="-1", help="Comma-separated token ids to suppress. '-1' means default behavior (no custom suppression).")
    g_dec.add_argument("--suppress_numerals", action="store_true", help="If set, suppress numeral tokens (may reduce digit-heavy hallucinations, but can remove useful numbers).")
    g_dec.add_argument("--initial_prompt", type=str, default=None, help="Optional initial prompt to steer style/vocabulary (e.g., domain terms).")
    g_dec.add_argument("--hotwords", type=str, default=None, help="Optional hotwords string to bias toward certain words (backend-dependent).")
    g_dec.add_argument("--condition_on_previous_text", type=_str2bool, default=False, help="If true, condition decoding on previous text (can improve continuity but may compound errors).")
    g_dec.add_argument("--fp16", type=_str2bool, default=True, help="Whether to use fp16 where applicable (mostly impacts GPU performance).")

    g_dec.add_argument("--temperature_increment_on_fallback", type=float, default=0.2, help="When decoding fails, gradually increase temperature by this increment (0 disables schedule).")
    g_dec.add_argument("--compression_ratio_threshold", type=float, default=2.4, help="Heuristic threshold for repetitive output; lower = more aggressive filtering.")
    g_dec.add_argument("--logprob_threshold", type=float, default=-1.0, help="Minimum average logprob threshold; raise to filter low-confidence segments (can drop content).")
    g_dec.add_argument("--no_speech_threshold", type=float, default=0.6, help="Probability threshold to treat a segment as silence; raise to skip more audio.")

    # ---------------------------
    # Subtitle formatting (mainly affects WhisperX segmentation style)
    # ---------------------------
    g_sub = ap.add_argument_group("Subtitle formatting / segmentation")

    g_sub.add_argument("--max_line_width", type=int, default=None, help="Max subtitle line width (characters). If set, subtitles will be wrapped more aggressively.")
    g_sub.add_argument("--max_line_count", type=int, default=None, help="Max subtitle line count. If set, subtitles will split into more segments.")
    g_sub.add_argument("--highlight_words", type=_str2bool, default=False, help="If true, enable word highlighting in subtitle-like outputs (backend-dependent).")
    g_sub.add_argument(
        "--segment_resolution",
        type=str,
        default="sentence",
        choices=["sentence", "chunk"],
        help="How to segment output: sentence-level tends to read better; chunk-level may follow VAD boundaries more closely.",
    )

    # ---------------------------
    # Misc
    # ---------------------------
    g_misc = ap.add_argument_group("Misc")

    g_misc.add_argument("--threads", type=int, default=0, help="CPU thread count hint. 0 = backend default.")
    g_misc.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (needed for some diarization/VAD models, especially pyannote).")
    g_misc.add_argument("--print_progress", type=_str2bool, default=False, help="Show progress bars / progress logs during processing.")
    g_misc.add_argument(
        "--verbose",
        type=_str2bool,
        default=False,
        help="Print backend info logs (model loading, alignment/diarization status, etc.).",
    )

    return ap


def _str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def run_cli() -> None:
    ap = _build_argparser()
    args = ap.parse_args()

    # Pull out orchestration-level args
    args_dict: Dict[str, Any] = vars(args).copy()
    inputs = args_dict.pop("inputs")
    backend = args_dict.pop("backend")
    output_dir = args_dict.pop("output_dir")
    run_name = args_dict.pop("run_name")
    output_format = args_dict.pop("output_format")
    save_subtitled_video = args_dict.pop("save_subtitled_video")

    # Remaining options are passed straight to backend / transcribe
    res = transcribe(
        inputs=inputs,
        backend=backend,
        output_dir=output_dir,
        run_name=run_name,
        output_format=output_format,
        save_subtitled_video=save_subtitled_video,
        **args_dict,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_cli()


__all__ = ["transcribe", "run_cli"]
