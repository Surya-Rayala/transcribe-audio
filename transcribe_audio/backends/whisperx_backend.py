

from __future__ import annotations

"""
WhisperX-backed ASR backend.

This is a thin wrapper around the `whisperx` library that:
  - Mirrors (most of) the official WhisperX CLI options.
  - Returns a normalized TranscriptResult in the `asr-v1` schema.
  - Leaves run_dir management and sidecar file writing to `audio_processing.asr`.

Notes:
  - This backend assumes `whisperx` is installed and importable.
  - We intentionally keep this minimal and opinionated for personal use,
    not as a fully generic wrapper for every upstream change.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    SCHEMA_VERSION,
    BaseASRBackend,
    Segment,
    TranscriptResult,
    Word,
)


try:
    import torch  # type: ignore
    import whisperx  # type: ignore
    from whisperx.diarize import DiarizationPipeline  # type: ignore
except Exception as e:  # pragma: no cover - soft failure, raised on use
    whisperx = None  # type: ignore
    DiarizationPipeline = None  # type: ignore
    _WHISPERX_IMPORT_ERROR = e


class WhisperXBackend(BaseASRBackend):
    """WhisperX implementation of BaseASRBackend."""

    def __init__(self, **config: Any) -> None:
        """
        Accepts a superset of WhisperX CLI-style options via **config.

        Most important keys (all optional here, defaults chosen to match upstream):
          - model: str = "small"
          - model_dir: Optional[str]
          - model_cache_only: bool
          - device: Optional[str] (None -> auto from torch)
          - device_index: int
          - batch_size: int
          - compute_type: str = "float16"
          - task: "transcribe" | "translate"
          - language: Optional[str]

          - align_model: Optional[str]
          - interpolate_method: str
          - no_align: bool
          - return_char_alignments: bool

          - vad_method, vad_onset, vad_offset, chunk_size

          - diarize: bool
          - min_speakers, max_speakers
          - diarize_model: str
          - speaker_embeddings: bool

          - temperature, best_of, beam_size, patience, length_penalty,
            suppress_tokens, suppress_numerals,
            initial_prompt, hotwords, condition_on_previous_text, fp16,
            temperature_increment_on_fallback,
            compression_ratio_threshold, logprob_threshold,
            no_speech_threshold

          - max_line_width, max_line_count, highlight_words, segment_resolution

          - threads, hf_token, print_progress
        """
        super().__init__(**config)
        self.backend_id = "whisperx"

    # Helper: choose device string similar to upstream defaults
    def _select_device(self, cfg: Dict[str, Any]) -> str:
        dev = cfg.get("device")
        if isinstance(dev, str) and dev:
            return dev
        # Fallback to upstream-style default
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

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
        if whisperx is None:  # pragma: no cover
            raise ImportError(
                "whisperx is required for WhisperXBackend but failed to import"
            ) from _WHISPERX_IMPORT_ERROR

        if not inputs:
            raise ValueError("No input files provided to WhisperXBackend.transcribe().")

        # Merge runtime kwargs over initial config
        cfg: Dict[str, Any] = dict(self.config)
        cfg.update(kwargs)

        # Explicit args (mirroring whisperx CLI defaults where sensible)
        model_name: str = cfg.get("model", "small")
        model_dir: Optional[str] = cfg.get("model_dir")
        model_cache_only: bool = bool(cfg.get("model_cache_only", False))
        device = self._select_device(cfg)
        device_index: int = int(cfg.get("device_index", 0))
        batch_size: int = int(cfg.get("batch_size", 8))
        compute_type: str = cfg.get("compute_type", "float16")

        # Behavior flags
        task = task or cfg.get("task", "transcribe")
        language = language or cfg.get("language")

        # Alignment / diarization / vad options (passed through where supported)
        no_align: bool = bool(cfg.get("no_align", False))
        align_model_name: Optional[str] = cfg.get("align_model")
        interpolate_method: str = cfg.get("interpolate_method", "nearest")
        return_char_alignments: bool = bool(cfg.get("return_char_alignments", False))

        diarize: bool = bool(cfg.get("diarize", False))
        min_speakers = cfg.get("min_speakers")
        max_speakers = cfg.get("max_speakers")
        diarize_model_name: str = cfg.get(
            "diarize_model", "pyannote/speaker-diarization-3.1"
        )
        speaker_embeddings: bool = bool(cfg.get("speaker_embeddings", False))

        vad_method: str = cfg.get("vad_method", "pyannote")
        vad_onset: float = float(cfg.get("vad_onset", 0.500))
        vad_offset: float = float(cfg.get("vad_offset", 0.363))
        chunk_size: int = int(cfg.get("chunk_size", 30))

        # Decoding / sampling parameters; we forward where whisperx exposes hooks.
        temperature: float = float(cfg.get("temperature", 0.0))
        best_of = cfg.get("best_of", 5)
        beam_size = cfg.get("beam_size", 5)
        patience: float = float(cfg.get("patience", 1.0))
        length_penalty: float = float(cfg.get("length_penalty", 1.0))

        suppress_tokens = cfg.get("suppress_tokens", "-1")
        suppress_numerals: bool = bool(cfg.get("suppress_numerals", False))
        initial_prompt: Optional[str] = cfg.get("initial_prompt")
        hotwords: Optional[str] = cfg.get("hotwords")
        condition_on_previous_text: bool = bool(
            cfg.get("condition_on_previous_text", False)
        )
        fp16: bool = bool(cfg.get("fp16", True))

        temperature_increment_on_fallback = cfg.get(
            "temperature_increment_on_fallback", 0.2
        )
        compression_ratio_threshold = cfg.get(
            "compression_ratio_threshold", 2.4
        )
        logprob_threshold = cfg.get("logprob_threshold", -1.0)
        no_speech_threshold = cfg.get("no_speech_threshold", 0.6)

        max_line_width = cfg.get("max_line_width")
        max_line_count = cfg.get("max_line_count")
        highlight_words: bool = bool(cfg.get("highlight_words", False))
        segment_resolution: str = cfg.get(
            "segment_resolution", "sentence"
        )

        threads = cfg.get("threads", 0)
        hf_token = cfg.get("hf_token")

        if threads and isinstance(threads, int) and threads > 0:
            # Let whisperx/torch respect this; no need for complex handling.
            try:
                torch.set_num_threads(int(threads))  # type: ignore[attr-defined]
            except Exception:
                pass

        verbose = bool(cfg.get("verbose", True))

        work_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load and transcribe with WhisperX ----

        # Normalize suppress_tokens into a list[int] like upstream CLI
        if isinstance(suppress_tokens, str):
            suppress_tokens_list: List[int] = []
            for tok in suppress_tokens.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        suppress_tokens_list.append(int(tok))
                    except ValueError:
                        # ignore non-int tokens for robustness
                        continue
            if not suppress_tokens_list:
                suppress_tokens_list = [-1]
        elif isinstance(suppress_tokens, (list, tuple)):
            suppress_tokens_list = [int(t) for t in suppress_tokens]
        else:
            suppress_tokens_list = [-1]

        # Handle temperature schedule similar to whisperx CLI
        if temperature_increment_on_fallback is not None:
            try:
                start_temp = float(temperature)
                inc = float(temperature_increment_on_fallback)
                if inc > 0:
                    temps = tuple(
                        t for t in
                        [start_temp + i * inc for i in range(int((1.0 - start_temp) / inc) + 1)]
                        if t <= 1.0 + 1e-6
                    )
                else:
                    temps = (start_temp,)
            except Exception:
                temps = (float(temperature),)
        else:
            temps = (float(temperature),)

        asr_options: Dict[str, Any] = {
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": patience,
            "length_penalty": length_penalty,
            "temperatures": temps,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
            "suppress_tokens": suppress_tokens_list,
            "suppress_numerals": suppress_numerals,
        }

        # Map our config onto whisperx.asr.load_model signature
        vad_options = {
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        }

        asr_model = whisperx.load_model(
            whisper_arch=model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            asr_options=asr_options,
            language=language,
            vad_method=vad_method,
            vad_options=vad_options,
            task=task,
            download_root=model_dir,
            local_files_only=bool(model_cache_only),
            threads=int(threads) if isinstance(threads, int) and threads > 0 else 4,
        )

        # For simplicity we currently support a single logical run:
        # if multiple inputs are passed, we transcribe them sequentially and
        # concatenate segments with monotonically increasing timestamps.
        all_segments: List[Segment] = []
        current_offset = 0.0

        # For metadata/debug
        raw_segments: List[Dict[str, Any]] = []

        for inp in inputs:
            audio_path = Path(inp)
            if not audio_path.exists():
                raise FileNotFoundError(f"Input not found: {audio_path}")

            audio = whisperx.load_audio(str(audio_path))

            if verbose:
                print(f"[whisperx] Transcribing: {audio_path}")

            # whisperx.asr_model.transcribe-style API
            asr_result = asr_model.transcribe(
                audio=audio,
                batch_size=batch_size,
                num_workers=0,
                language=language,
                task=task,
                chunk_size=chunk_size,
                print_progress=print_progress,
                combined_progress=False,
                verbose=verbose,
            )

            segments = asr_result.get("segments", []) or []
            language_out = asr_result.get("language", language)

            # Optional alignment
            if not no_align:
                if verbose:
                    print("[whisperx] Running alignment...")
                try:
                    # whisperx.alignment.load_align_model(language_code, device, model_name=None, model_dir=None)
                    align_model, align_meta = whisperx.load_align_model(
                        language_code=language_out,
                        device=device,
                        model_name=align_model_name,
                        model_dir=model_dir,
                    )

                    aligned = whisperx.align(
                        segments,
                        align_model,
                        align_meta,
                        audio,
                        device,
                        interpolate_method=interpolate_method,
                        return_char_alignments=return_char_alignments,
                        print_progress=print_progress,
                    )

                    # align() returns a dict with "segments" and "word_segments"
                    segments = aligned.get("segments", segments) or segments
                except Exception as e:  # pragma: no cover - alignment is optional
                    if verbose:
                        print(f"[whisperx] Alignment failed ({e}); continuing without.")

            # Optional diarization
            if diarize:
                if verbose:
                    print("[whisperx] Running diarization...")
                try:
                    diarize_model = DiarizationPipeline(
                        model_name=diarize_model_name,
                        use_auth_token=hf_token,
                        device=device,
                    )

                    # Call diarization with speaker constraints if provided
                    if speaker_embeddings:
                        diarize_df, embeddings = diarize_model(
                            audio,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                            return_embeddings=True,
                        )
                    else:
                        diarize_df = diarize_model(
                            audio,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                            return_embeddings=False,
                        )
                        embeddings = None

                    # assign_word_speakers expects a transcript-like dict with "segments"
                    transcript_like = {"segments": segments}
                    transcript_like = whisperx.assign_word_speakers(
                        diarize_df,
                        transcript_like,
                        speaker_embeddings=embeddings,
                        fill_nearest=False,
                    )
                    segments = transcript_like.get("segments", segments)
                except Exception as e:  # pragma: no cover
                    if verbose:
                        print(f"[whisperx] Diarization failed ({e}); continuing without.")
                    diarize = False

            # Normalize segments into asr-v1 schema, applying running offset
            for seg in segments:
                start = float(seg.get("start", 0.0)) + current_offset
                end = float(seg.get("end", start))
                text = str(seg.get("text", "")).strip()
                speaker = seg.get("speaker")

                # Word-level
                words_data = seg.get("words") or []
                words: List[Word] = []
                for w in words_data:
                    w_start = float(w.get("start", start)) + current_offset
                    w_end = float(w.get("end", w_start))
                    w_text = str(w.get("word", w.get("text", "")))
                    w_prob = w.get("probability")
                    w_speaker = w.get("speaker", speaker)
                    words.append(
                        Word(
                            start=w_start,
                            end=w_end,
                            text=w_text,
                            prob=w_prob,
                            speaker=w_speaker,
                        )
                    )

                seg_out: Segment = Segment(
                    start=start,
                    end=end,
                    text=text,
                )
                if speaker is not None:
                    seg_out["speaker"] = str(speaker)
                if words:
                    seg_out["words"] = words

                all_segments.append(seg_out)
                raw_segments.append(seg)

            # Advance offset so next file starts after this one.
            if segments:
                last_end = float(segments[-1].get("end", 0.0))
                current_offset += max(0.0, last_end)

        # Build metadata summary
        meta: Dict[str, Any] = {
            "model": model_name,
            "device": device,
            "device_index": device_index,
            "compute_type": compute_type,
            "task": task,
            "language": language,
            "vad_method": vad_method,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
            "chunk_size": chunk_size,
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "align": not no_align,
            "align_model": align_model_name,
            "interpolate_method": interpolate_method,
            "return_char_alignments": return_char_alignments,
            "speaker_embeddings": speaker_embeddings,
            "highlight_words": highlight_words,
            "segment_resolution": segment_resolution,
            "suppress_tokens": suppress_tokens,
            "suppress_numerals": suppress_numerals,
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
            "condition_on_previous_text": condition_on_previous_text,
            "fp16": fp16,
            "temperature": temperature,
            "best_of": best_of,
            "beam_size": beam_size,
            "patience": patience,
            "length_penalty": length_penalty,
            "temperature_increment_on_fallback": temperature_increment_on_fallback,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "threads": threads,
            "hf_token_used": bool(hf_token),
            "raw_segments_count": len(raw_segments),
        }

        # Build TranscriptResult
        result: TranscriptResult = {
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend_id,
            "run_name": str(self.config.get("run_name", "")) or "",
            "inputs": [str(p) for p in inputs],
            "work_dir": str(work_dir),
            "language": language,
            "task": task,
            "diarization": bool(diarize),
            "alignment": bool(not no_align),
            "segments": all_segments,
            "metadata": meta,
        }

        return result


__all__ = ["WhisperXBackend"]