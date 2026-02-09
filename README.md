# transcribe-audio

A CLI + Python package for automatic speech recognition (ASR) on **audio or video** using pluggable backends (currently **WhisperX**).

This tool:
- Accepts **audio** (e.g., `wav/mp3/m4a`) and **video** (e.g., `mp4/mkv/mov`)
- Automatically extracts audio from video inputs (requires **ffmpeg**)
- Writes a canonical `transcript.json` plus optional sidecars: `txt/srt/vtt/tsv`
- Optionally creates a **subtitled MP4** with burned-in subtitles

---

## ffmpeg requirement

ffmpeg is required for:
- extracting audio from **video** inputs
- creating **subtitled videos**

Install examples:
- macOS: `brew install ffmpeg`
- Conda env (recommended inside conda): `conda install -c conda-forge ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`

---

## Install with `pip`

### Install

```bash
pip install transcribe-audio
```

> If you haven’t published to PyPI yet, replace `transcribe-audio` above with your final package name.

---

## CLI usage

### Global help

```bash
python -m transcribe-audio.transcribe -h
# or
python -m transcribe-audio.transcribe --help
```

### List supported backends

```bash
python -c "from transcribe-audio.backends import available_backends; print('\\n'.join(available_backends()))"
```

### Basic transcription (ALL formats + subtitled video)

```bash
python -m transcribe-audio.transcribe \
  -i in.wav \
  --backend whisperx \
  --model small \
  --task transcribe \
  -o out \
  --run-name demo_transcribe \
  -f all \
  --save-subtitled-video
```

### Transcription + diarization (ALL formats + subtitled video)

```bash
python -m transcribe-audio.transcribe \
  -i in.wav \
  --backend whisperx \
  --model small \
  --task transcribe \
  -o out \
  --run-name demo_diarize \
  -f all \
  --diarize \
  --hf_token "$HF_TOKEN" \
  --save-subtitled-video
```

Outputs land in:

```text
out/<run-name>/
  transcript.json
  transcript.txt
  transcript.srt
  transcript.vtt
  transcript.tsv
  *_subtitled.mp4
```

---

## Hugging Face token + gated model access (needed for diarization)

Diarization models are often gated (require accepting terms) and may require a Hugging Face token.

**When is an HF token needed?**
- **Usually only when models are not already cached** (first download, cache miss, or when the library checks the Hub).
- For **gated** repos (common for diarization), a token + acceptance is required for downloads.
- If you run fully offline with everything cached (and `--model_cache_only true` for ASR), a token is often not needed.

### Create a token (read-only)

- Go to HF settings → Access Tokens:
  https://huggingface.co/settings/tokens
- Create a New token with Role: **Read**
- Copy the token (starts with `hf_...`)

Docs (tokens):
https://huggingface.co/docs/hub/en/security-tokens

### Accept gated model access

When you use diarization, you may be prompted to accept access for pyannote models.
Open the model page mentioned in your errors (or the default diarization pipeline) and click “Agree and access”.

Browse diarization models:
https://huggingface.co/pyannote/models

Common default diarization pipeline used by WhisperX:
https://huggingface.co/pyannote/speaker-diarization-3.1

### Recommended: store token in an environment variable

```bash
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXX"
```

Then pass it to the CLI:

```bash
--hf_token "$HF_TOKEN"
```

Security note: rotate/revoke tokens you’ve pasted anywhere public.

---

## Arguments

### Core I/O

- `--input`, `-i <path ...>` (required): One or more audio/video files.
  - Video inputs are auto-converted to WAV via ffmpeg into the run directory.
- `--output-dir`, `-o <dir>`: Root output directory (default `out`).
- `--run-name <name>`: Run folder name under `--output-dir`.
  - If omitted, derived as `<first_input_stem>_<backend>`.
- `--output-format`, `-f <format>`: Transcript sidecars to generate:
  - `json` (canonical output, always written)
  - `txt` (plain text, includes speaker prefix when available)
  - `srt` / `vtt` (subtitle formats)
  - `tsv` (tab-separated: start/end/speaker/text)
  - `all` (json + txt + srt + vtt + tsv)
- `--save-subtitled-video [out.mp4]`: Create an MP4 with burned-in subtitles.
  - If a video input exists: subtitles burned onto the first video.
  - If audio-only: creates a small black-strip video with subtitles.
  - With no filename, auto-names output inside the run folder.

### Backend selection

- `--backend <name>`: Which backend to use (choices come from `available_backends()`).
  - Current default: `whisperx`.

### Model & performance

- `--model <name>`: Whisper/faster-whisper model name (default `small`).
  - Larger models are more accurate but slower and require more memory.
  - Model list/reference: https://huggingface.co/collections/Systran/faster-whisper
- `--model_cache_only true|false`: If true, do not download models (offline mode).
  - Will fail if models are not already cached.
- `--model_dir <path>`: Directory for model downloads/cache.
- `--device <cpu|cuda|...>`: Device override. If omitted, backend auto-selects.
- `--device_index <int>`: GPU index (default 0).
- `--batch_size <int>`: Decoding batch size (default 8).
  - Higher can be faster but uses more VRAM. Lower if you hit OOM.
- `--compute_type <float16|float32|int8>`:
  - `float16`: faster on GPU, uses less VRAM (may be less stable on some setups)
  - `float32`: safest/stable
  - `int8`: lower memory, can be good on CPU; may impact accuracy

### Task & language

- `--task <transcribe|translate>`:
  - `transcribe`: speech → text in the same language
  - `translate`: speech → English (if supported by the model)
- `--language <code>`: Force language (e.g., `en`, `es`).
  - If omitted, auto-detect is used (slower but convenient).

### Alignment (word-level timestamps)

- `--no_align`: Disable alignment.
  - Faster, but timestamps may be less precise and word timings may be missing.
- `--align_model <name>`: Optional alignment model override (usually auto-selected).
  - Alignment model reference list:
    https://docs.pytorch.org/audio/0.12.0/pipelines.html#wav2vec-2-0-hubert-fine-tuned-asr
- `--interpolate_method <nearest|linear|ignore>`:
  - `nearest`: fill missing word times using nearest neighbors (safe default)
  - `linear`: interpolate missing times
  - `ignore`: leave missing (can create gaps)
- `--return_char_alignments`: Include character-level alignment (more detail, heavier output).

### VAD (speech detection & chunking)

- `--vad_method <pyannote|silero>`: Voice activity detection backend.
  - Changes segmentation and can affect diarization quality.
- `--vad_onset <float>`: Speech-start threshold.
  - Higher = fewer false positives, but may miss quiet speech.
- `--vad_offset <float>`: Speech-end threshold.
  - Higher = cuts off sooner; lower = keeps trailing audio longer.
- `--chunk_size <seconds>`: Chunk size used in transcription (default 30).
  - Smaller chunks can reduce memory usage; larger can be faster.

### Diarization (speaker labels)

- `--diarize`: Enable speaker diarization.
  - Requires HF token + accepted model access in many setups.
  - Browse diarization models: https://huggingface.co/pyannote/models
- `--min_speakers <int>`: Optional lower bound on speakers.
- `--max_speakers <int>`: Optional upper bound on speakers.
- `--diarize_model <repo>`: Diarization pipeline to use.
  - Default: `pyannote/speaker-diarization-3.1`
- `--speaker_embeddings`: Compute/return speaker embeddings.
  - Can improve speaker consistency but increases compute/memory.

### Decoding / advanced ASR knobs (advanced)

These knobs mainly affect ASR decoding and stability:

- `--temperature <float>`: 0.0 is deterministic; higher may help difficult audio but can introduce errors.
- `--best_of <int>`: More candidates; slower; sometimes better.
- `--beam_size <int>`: Larger beam can be more accurate; slower.
- `--patience <float>`: Beam search patience; higher explores more; slower.
- `--length_penalty <float>`: Penalizes/encourages longer outputs.
- `--suppress_tokens <csv>`: Token IDs to suppress (-1 = default behavior).
- `--suppress_numerals`: Suppress numeral tokens (may reduce number hallucinations but can remove real numbers).
- `--initial_prompt <text>`: Steer style/vocabulary (domain phrases).
- `--hotwords <text>`: Bias toward certain words (backend-dependent).
- `--condition_on_previous_text true|false`: Better continuity, but can compound errors.
- `--fp16 true|false`: Prefer fp16 when supported (mostly affects GPU).

Fallback / thresholds:
- `--temperature_increment_on_fallback <float>`: Increase temperature when decoding fails.
- `--compression_ratio_threshold <float>`: Lower filters repetitive output more aggressively.
- `--logprob_threshold <float>`: Increase to drop low-confidence segments (can remove content).
- `--no_speech_threshold <float>`: Raise to skip more silence/background.

### Subtitle formatting / segmentation

- `--max_line_width <int>`: Wrap subtitle lines to this width.
- `--max_line_count <int>`: Limit subtitle lines per cue.
- `--highlight_words true|false`: Word highlighting in subtitle-like outputs (backend-dependent).
- `--segment_resolution <sentence|chunk>`:
  - `sentence`: generally more readable subtitle segments
  - `chunk`: follows chunk/VAD boundaries more closely

### Misc

- `--threads <int>`: CPU thread hint (0 = backend default).
- `--hf_token <token>`: Hugging Face token for gated models (diarization, some VAD).
- `--print_progress true|false`: Show progress bars/logs.
- `--verbose true|false`: Print backend info logs and warnings (recommended while setting up).

---

## Python usage (import)

After install:

```bash
python -c "from transcribe-audio import transcribe; print(transcribe)"
```

### Simple transcription from Python

```python
from transcribe-audio import transcribe

res = transcribe(
    inputs="in.wav",
    backend="whisperx",
    model="small",
    output_format="all",
    output_dir="out",
    run_name="py_demo",
    save_subtitled_video="__AUTO__",  # or "my_subs.mp4"
)

print("Run dir:", res["run_dir"])
print("JSON:", res["outputs"]["json"])
```

### Transcription + diarization from Python

```python
from transcribe-audio import transcribe
import os

res = transcribe(
    inputs="in.wav",
    backend="whisperx",
    model="small",
    output_format="all",
    output_dir="out",
    run_name="py_diarize",
    save_subtitled_video="__AUTO__",
    diarize=True,
    hf_token=os.environ.get("HF_TOKEN"),
)

print(res["outputs"])
```

---

## Install from GitHub (uv)

Install uv (Astral):
https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Verify:

```bash
uv --version
```

Clone + install deps:

```bash
git clone https://github.com/Surya-Rayala/transcribe-audio.git
cd transcribe-audio
uv sync
```

CLI help (uv environment):

```bash
uv run python -m transcribe-audio.transcribe -h
```

Basic transcription (uv):

```bash
uv run python -m transcribe-audio.transcribe \
  -i in.wav \
  --backend whisperx \
  --model small \
  --task transcribe \
  -o out \
  --run-name demo_transcribe \
  -f all \
  --save-subtitled-video
```

Transcription + diarization (uv):

```bash
uv run python -m transcribe-audio.transcribe \
  -i in.wav \
  --backend whisperx \
  --model small \
  --task transcribe \
  -o out \
  --run-name demo_diarize \
  -f all \
  --diarize \
  --hf_token "$HF_TOKEN" \
  --save-subtitled-video
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.