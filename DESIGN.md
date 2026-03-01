# System Design — Rangoli (Podcast Transcriber)

**Version 1.0.0** | **Author: Gaurav Mathur** | **License: MIT** | **2026**

## Overview

Rangoli is a local-first podcast transcription tool that converts podcast audio to text using OpenAI Whisper or faster-whisper, with optional speaker diarization. It ships as two interfaces — a CLI for scripting and a GUI for interactive use — sharing a common transcription pipeline but with separate data persistence strategies.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interfaces                     │
│                                                         │
│  ┌──────────────────┐       ┌────────────────────────┐  │
│  │  CLI              │       │  GUI (CustomTkinter)   │  │
│  │  podcast_          │       │  podcast_gui.py        │  │
│  │  transcriber.py   │       │                        │  │
│  │                   │       │  ┌──────────────────┐  │  │
│  │  argparse flags   │       │  │ Sidebar          │  │  │
│  │  stdout output    │       │  │  Model dropdown  │  │  │
│  │  .txt file output │       │  │  Engine dropdown │  │  │
│  │                   │       │  │  Diarize toggle  │  │  │
│  └────────┬──────────┘       │  │  Podcast list    │  │  │
│           │                  │  └──────────────────┘  │  │
│           │                  │  ┌──────────────────┐  │  │
│           │                  │  │ Main area        │  │  │
│           │                  │  │  Episode table   │  │  │
│           │                  │  │  Progress bar    │  │  │
│           │                  │  │  Progress detail │  │  │
│           │                  │  │  Transcript view │  │  │
│           │                  │  └──────────────────┘  │  │
│           │                  └───────────┬────────────┘  │
└───────────┼──────────────────────────────┼──────────────┘
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────┐
│                  Shared Components                      │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ RSS Fetcher  │  │ Transcription│  │ Diarization  │  │
│  │ feedparser + │  │ Engine       │  │ pyannote     │  │
│  │ requests     │  │ Abstraction  │  │ .audio       │  │
│  └──────────────┘  └──────┬───────┘  └──────────────┘  │
│                           │                             │
│              ┌────────────┼────────────┐                │
│              ▼                         ▼                │
│  ┌──────────────────┐     ┌──────────────────────┐      │
│  │ OpenAI Whisper   │     │ faster-whisper        │      │
│  │ whisper.load_    │     │ WhisperModel(         │      │
│  │   model()        │     │   compute_type=int8)  │      │
│  │ FP32, blocking   │     │ CTranslate2, lazy     │      │
│  └──────────────────┘     └──────────────────────┘      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Speaker Assignment                               │   │
│  │ Overlap-based segment-to-speaker mapping          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
            │                              │
            ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│ .txt files       │          │ SQLite (database.py) │
│ (CLI output)     │          │ (GUI persistence)    │
└──────────────────┘          └──────────────────────┘
```

## Component Details

### 1. RSS Feed Parser

**Files**: `podcast_transcriber.py:fetch_feed()`, `podcast_gui.py:fetch_feed()`

Both interfaces have their own `fetch_feed()` — the CLI version calls `sys.exit()` on error, the GUI version raises exceptions for the UI thread to catch.

**Audio URL discovery** uses a 3-tier fallback:
1. `<enclosure>` tags with `audio/*` MIME type (standard RSS podcasts)
2. `<media:content>` elements (Media RSS extension)
3. Links with audio file extensions (`.mp3`, `.m4a`, `.wav`, `.ogg`, `.aac`)

**SSL handling**: macOS Python may lack proper SSL certificates. Both entry points set `SSL_CERT_FILE` to `certifi`'s CA bundle and override `ssl._create_default_https_context` to fix this at import time.

### 2. Transcription Engine Abstraction

**Files**: `podcast_transcriber.py:transcribe_audio()`, `podcast_gui.py:_transcribe_worker()`

The engine abstraction supports two backends selected at runtime:

#### OpenAI Whisper (`engine="whisper"`)
- **Import**: `import whisper` (required, always available)
- **Model loading**: `whisper.load_model(model_name)` — downloads to `~/.cache/whisper/` on first use
- **Transcription**: `model.transcribe(path, verbose=True, fp16=False)` with stdout redirected to `_WhisperProgressWriter`
- **Compute**: FP32 on CPU (FP16 explicitly disabled to avoid runtime warnings on CPU-only systems)
- **Output**: Returns `dict` with `"segments"` list and `"text"` string
- **Progress**: When `verbose=True`, Whisper prints each segment as `[MM:SS.mmm --> MM:SS.mmm] text`. `_WhisperProgressWriter` captures stdout, parses these timestamps via regex, and pushes per-segment progress updates identical to faster-whisper. Audio duration is obtained via `whisper.audio.load_audio()` / `whisper.audio.SAMPLE_RATE`

#### faster-whisper (`engine="faster-whisper"`)
- **Import**: Conditional `from faster_whisper import WhisperModel` with `FASTER_WHISPER_AVAILABLE` flag
- **Model loading**: `WhisperModel(model_name, device="cpu", compute_type="int8")` — downloads from Hugging Face Hub
- **Transcription**: `model.transcribe(path)` returns `(segment_iterator, TranscriptionInfo)`
- **Compute**: int8 quantization via CTranslate2 — ~4x faster than OpenAI Whisper on CPU
- **Output**: Lazy generator yielding `Segment` dataclasses (converted to dicts for uniform handling)
- **Behavior**: Segments are yielded incrementally as processed. `TranscriptionInfo.duration` provides total audio length immediately, enabling real-time progress: `segment.end / info.duration`

#### Engine selection
- **CLI**: `--engine` flag, defaults to `WHISPER_ENGINE` env var, falls back to `"whisper"`
- **GUI**: Sidebar dropdown, populated from `WHISPER_ENGINES` list. If `faster-whisper` is not installed, the dropdown only shows `["whisper"]`

#### Segment format normalization
Both engines produce segments normalized to: `{"start": float, "end": float, "text": str}`. This uniform format flows into diarization and formatting without branching.

### 3. Speaker Diarization

**Files**: `podcast_transcriber.py:diarize_audio()`, `podcast_transcriber.py:assign_speakers()`

**Pipeline**: pyannote.audio's `speaker-diarization-3.1` pretrained pipeline, loaded via Hugging Face token.

**Speaker-to-segment assignment algorithm**:
```
For each Whisper segment [seg_start, seg_end]:
    For each diarization turn [turn_start, turn_end, speaker_id]:
        overlap = max(0, min(seg_end, turn_end) - max(seg_start, turn_start))
    Assign speaker with maximum overlap to the segment
    Map raw speaker IDs (SPEAKER_00, SPEAKER_01) → friendly names (Speaker 1, Speaker 2)
```

This is an O(S * T) algorithm where S = number of Whisper segments and T = number of diarization turns. For typical podcast episodes (hundreds of segments, similar number of turns), this completes in milliseconds.

**Failure modes** (graceful degradation):
- `pyannote.audio` not installed → skips with warning, produces transcript without speakers
- `HF_TOKEN` missing/placeholder → skips with warning
- GUI checkbox disabled if `pyannote.audio` not available

### 4. GUI Application

**File**: `podcast_gui.py`

#### Application Identity

| Property   | Value              |
|------------|--------------------|
| Name       | Rangoli              |
| Version    | 1.0.0              |
| Author     | Gaurav Mathur       |
| License    | MIT                |
| Year       | 2026               |
| Icon       | `icon.png` (rangoli)|

Constants `APP_NAME`, `APP_VERSION`, `APP_AUTHOR`, `APP_YEAR` are defined at module level and used throughout the GUI.

On macOS, the menu bar app name is set to "Rangoli" by modifying `CFBundleName` in the main bundle's info dictionary via CoreFoundation C API (`ctypes`). This runs **before** `import tkinter` to ensure Tk reads the modified name when it initializes. No external dependencies required — uses only macOS system libraries. The window/dock icon is set via `iconphoto()`. Both are wrapped in try/except and degrade silently on failure.

#### Menu Bar

```
macOS:
  Rangoli │ File
  ──────┼──────────────────
  About │ Add Podcast  ⌘N
  Rangoli │ Remove Podcast
        │
        │ (Quit added by macOS automatically)

Other platforms:
  File             │ Help
  ─────────────────┼──────────
  Add Podcast Ctrl+N│ About Rangoli
  Remove Podcast   │
  ─────────────────│
  Quit       Ctrl+Q│
```

The About dialog displays the app icon, name, version, subtitle ("Podcast Transcriber"), author, copyright, and license.

#### Window Layout
```
┌──────────────────┬──────────────────────────────────────────┬───────────────┐
│  Sidebar (resiz) │  Main Area                               │  Transcript   │
│  PanedWindow     │  expands with window                     │  Panel        │
│                  │                                          │               │
│ [🪷 Rangoli  [+]]│  [Episode Title        < > 1/3  [Trans]] │  Transcript   │
│                  │                                          │  ──────────── │
│  Model:  [base▼] │  # │ Title     │Published│ Dur   │Status│  [00:00:05]   │
│  Engine: [wh..▼] │  1 │ Ep One    │Feb 24   │45m 0s │ Done │  Welcome to.. │
│  [✓] Diarize    │  2 │ Ep Two    │Feb 17   │30m 0s │      │  [00:00:12]   │
│                  │  3 │ Ep Three  │Feb 10   │52m 0s │      │  Thanks for.. │
│  ┌────────────┐  │                                          │               │
│  │ Podcast A  │  │  [████████████████░░░░] Stage 3/4: ...  │  [Copy][Close]│
│  │ Podcast B  │  │  faster-whisper 'base' — 42 segs | ...  │               │
│  │ Podcast C  │  │                                          │               │
│  │            │  │                                          │               │
│  └────────────┘  │                                          │               │
│  [Remove Podcast]│                                          │               │
└──────────────────┴──────────────────────────────────────────┴───────────────┘

Right-click a podcast → modal dialog with artwork, name, and description.
```

#### Duration Normalization

Episode durations from RSS feeds arrive in varied formats from the `itunes_duration` tag:
- Raw seconds: `"3600"` → `"1h 0m 0s"`
- MM:SS: `"45:30"` → `"45m 30s"`
- HH:MM:SS: `"1:23:45"` → `"1h 23m 45s"`
- Short: `"30"` → `"30s"`

`_normalize_duration()` converts all formats to a compact human-readable display. Zero-value leading components (hours, minutes) are omitted. Unparseable values are passed through unchanged.

#### Layout Structure
```
PanedWindow (horizontal, resizable sash):
  Pane 0 — Sidebar (initial 280px, min 200px):
    Row 0: Header (icon + "Rangoli" + add button)
    Row 1: Settings grid (Model, Engine, Diarize — aligned labels/dropdowns)
    Row 3: Scrollable podcast list (weight=1, expands)
    Row 4: Remove Podcast button

  Pane 1 — Main Area (min 500px):
    Row 0: Top bar (episode header, pagination, transcribe button)
    Row 1: Pane (weight=1, expands):
      Pane Row 0: Episode table with #/Title/Published/Duration/Status (weight=1)
      Pane Row 1: Progress area (fixed height)

  Pane 2 — Transcript Panel (optional, shown on episode select/transcription, min 250px):
    Row 0: "Transcript" header
    Row 1: Transcript textbox (scrollable, read-only, Menlo font)
    Row 2: Bottom bar (Copy + Close buttons)

  Podcast Info — Modal dialog (CTkToplevel, shown on right-click):
    Artwork (48x48) + podcast name, separator, description textbox, Close button
```

#### Threading Model

All network I/O and transcription runs in a daemon thread spawned from `_on_transcribe()`. The GUI thread never blocks.

**Thread communication**: The worker thread calls `self.after(0, lambda: ...)` to schedule UI updates on the main thread. This is Tkinter's standard thread-safe callback mechanism.

**Concurrency guard**: `self._transcribing` flag prevents:
- Starting a second transcription
- Selecting a different episode during transcription

#### Cancellation

Cancellation uses cooperative signaling via `threading.Event`:

- **Signal**: `self._cancel_event` (`threading.Event`) — set by `_on_stop()` on the main thread
- **Check**: `_check_cancelled()` — called by the worker at every interruptible point, raises `_TranscriptionCancelled` if the event is set
- **UI**: When transcription starts, the "Transcribe" button becomes a red "Stop" button. Clicking it sets the cancel event and changes the button text to "Stopping...". After the worker exits, `_transcription_done()` restores the button to its original state.

**Cancellation checkpoints** in `_transcribe_worker()`:
1. After each download chunk (8 KB granularity — stops within milliseconds)
2. After download completes, before model loading
3. After model loading completes, before transcription
4. After each faster-whisper segment (real-time cancellation during transcription)
5. After each OpenAI Whisper segment — `_WhisperProgressWriter.write()` checks the cancel event and raises `_TranscriptionCancelled` from within Whisper's `print()` call, aborting transcription mid-segment
6. Before and after diarization pipeline runs

**Exception flow**:
```
_check_cancelled() raises _TranscriptionCancelled
    │
    ▼
Inner try/finally block catches it:
    → Cleans up temp audio file (always)
    │
    ▼
Outer except _TranscriptionCancelled:
    → Sets progress to "Cancelled"
    → Shows "Transcription cancelled." in transcript box
    │
    ▼
Outer finally:
    → Calls _transcription_done() → restores button
```

**Uniform cancellation**: Both engines support per-segment cancellation. faster-whisper yields control back to Python after each segment via its lazy generator. OpenAI Whisper prints each segment to stdout when `verbose=True`; `_WhisperProgressWriter` intercepts these `print()` calls and checks the cancel event, raising `_TranscriptionCancelled` to abort transcription at the next segment boundary.

```
Main Thread                    Worker Thread
    │                               │
    ├─ _on_transcribe()             │
    │   set _transcribing=True      │
    │   clear _cancel_event         │
    │   button → red "Stop"         │
    │   spawn thread ──────────────►│
    │                               ├─ download audio
    │   ◄── after(0, update_prog) ──┤   _check_cancelled() per chunk
    │                               │
    ├─ user clicks Stop             ├─ _check_cancelled() → raises
    │   _cancel_event.set()         │   _TranscriptionCancelled
    │   button → "Stopping..."      │
    │                               ├─ finally: delete temp file
    │   ◄── after(0, "Cancelled") ──┤
    │   ◄── after(0, done) ─────────┤
    │   button → "Transcribe"       ▼
    │
    │  ── OR (normal completion) ──
    │                               │
    │                               ├─ load model
    │   ◄── after(0, update_prog) ──┤   _check_cancelled()
    │                               ├─ transcribe
    │   ◄── after(0, update_prog) ──┤   _check_cancelled() per segment
    │                               ├─ diarize (optional)
    │   ◄── after(0, update_prog) ──┤   _check_cancelled()
    │                               ├─ format + save to DB
    │   ◄── after(0, set_text) ─────┤
    │   ◄── after(0, done) ─────────┤
    │   button → "Transcribe"       ▼
```

#### Progress Pipeline

The progress bar maps the transcription pipeline to a 0.0-1.0 range across 4 stages:

| Range       | Stage                  | Detail line content                                                        |
|-------------|------------------------|----------------------------------------------------------------------------|
| 0.00 - 0.30 | 1/4: Download          | `3.2 / 7.1 MB (45%) — ~2m 15s remaining`                                  |
| 0.30 - 0.35 | 2/4: Load model        | `faster-whisper 'base' (int8, cpu)`                                        |
| 0.35 - 0.85 | 3/4: Transcribe        | `<engine> '<model>' — 42 segments \| 03:24 / 15:30 — ~3m 10s remaining`  |
| 0.85 - 0.95 | 4/4: Diarize           | `pyannote.audio — analyzing speakers (elapsed 45s)...`                     |
| 0.95 - 1.00 | Formatting             | (empty)                                                                    |
| 1.00        | Done                   | `Total time: 5m 32s`                                                       |

#### Time Estimation

**ETA calculation** (`_estimate_remaining(elapsed, fraction_done)`):
- Uses linear extrapolation: `remaining = elapsed / fraction_done * (1 - fraction_done)`
- Suppressed until `elapsed >= 2s` and `fraction_done > 0` to avoid wild early estimates
- Returns `None` (hidden) when remaining time is under 1 second

**Where ETAs are shown**:
- **Download**: ETA based on bytes downloaded / total bytes — updates per 8 KB chunk
- **faster-whisper transcription**: ETA based on `seg.end / info.duration` — updates per segment
- **Completion messages**: Show actual elapsed time (`in 3m 42s`) instead of ETA
- **Blocking stages** (model load, diarization): Show elapsed time only since no progress fraction is available
- **Cancellation**: Shows time elapsed before stop (`Stopped after 1m 23s`)
- **Final Done**: Shows total wall-clock time for the entire job

**Uniform progress across engines**: Both engines provide identical per-segment progress updates:
- **faster-whisper**: The segment generator is iterated one segment at a time. After each segment, progress is computed as `0.35 + (seg.end / info.duration) * 0.50`. `info.duration` is available immediately from `TranscriptionInfo`.
- **OpenAI Whisper**: `model.transcribe(verbose=True)` is called with `sys.stdout` redirected to `_WhisperProgressWriter`. The writer parses Whisper's `[MM:SS.mmm --> MM:SS.mmm] text` output via regex (`_WHISPER_TS_RE`), extracts the end timestamp, and computes progress as `0.35 + (end_time / audio_duration) * 0.50`. Audio duration is pre-computed via `whisper.audio.load_audio()` / `whisper.audio.SAMPLE_RATE`. Both engines show the same detail format: segment count, audio time processed vs total, and a live ETA.

### 5. Database Layer

**File**: `database.py`

#### Schema
```sql
podcasts
  id          INTEGER PRIMARY KEY AUTOINCREMENT
  rss_url     TEXT UNIQUE NOT NULL
  title       TEXT NOT NULL
  author      TEXT DEFAULT ''
  description TEXT DEFAULT ''
  image_url   TEXT DEFAULT ''
  added_at    TEXT NOT NULL              -- ISO 8601

episodes
  id          INTEGER PRIMARY KEY AUTOINCREMENT
  podcast_id  INTEGER NOT NULL           -- FK → podcasts.id (CASCADE DELETE)
  title       TEXT NOT NULL
  published   TEXT DEFAULT ''
  summary     TEXT DEFAULT ''
  audio_url   TEXT NOT NULL
  duration    TEXT DEFAULT ''

transcripts
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  episode_id      INTEGER UNIQUE NOT NULL -- FK → episodes.id (CASCADE DELETE)
  text            TEXT NOT NULL
  model_used      TEXT NOT NULL           -- e.g. "base", "medium"
  diarized        INTEGER DEFAULT 0       -- boolean
  transcribed_at  TEXT NOT NULL           -- ISO 8601
```

#### Key design decisions
- **`PRAGMA foreign_keys = ON`**: Enforced per-connection (SQLite default is OFF). Enables cascade deletes.
- **CASCADE DELETE**: Deleting a podcast automatically removes all its episodes and their transcripts.
- **`episode_id UNIQUE` on transcripts**: One transcript per episode. Re-transcribing replaces via `INSERT OR REPLACE`.
- **`rss_url UNIQUE` on podcasts**: Prevents duplicate podcast entries.
- **No connection pooling**: Each operation opens/closes its own connection. SQLite handles this efficiently for a single-user desktop app.
- **`row_factory = sqlite3.Row`**: Returns dict-like rows for convenient field access.

#### Entity Relationship
```
podcasts 1──────* episodes 1──────? transcripts
         CASCADE          CASCADE
```

### 6. CLI Application

**File**: `podcast_transcriber.py`

**Flow**:
```
parse args
    │
    ├─ --list ──────► fetch_feed() → list_episodes() → exit
    │
    └─ --episode N ─► fetch_feed() → transcribe_episode()
                          │
                          ├─ download_audio()      → temp file
                          ├─ transcribe_audio()     → segments (whisper or faster-whisper)
                          ├─ diarize_audio()        → speaker timeline (optional)
                          ├─ assign_speakers()      → annotated segments
                          ├─ format_transcript()    → text
                          └─ write .txt file        → output dir
```

The CLI shares no code with the GUI (both have their own `fetch_feed()` and formatting logic). This is intentional — the CLI uses `sys.exit()` for error handling and writes plain files, while the GUI uses exceptions and SQLite.

## Data Flow

### Transcription Pipeline (both interfaces)

```
RSS Feed URL
    │
    ▼
feedparser ──► Episode list (title, audio_url, duration, ...)
                    │
                    ▼ (user selects episode)
              Download audio ──► temp file (.mp3/.m4a/...)
                    │
                    ▼
              Engine dispatch ─┬─► whisper.load_model() + transcribe()
                               │   Returns: {"segments": [...], "text": "..."}
                               │
                               └─► WhisperModel() + transcribe()
                                   Returns: (lazy_segments, TranscriptionInfo)
                                   Materialized to: {"segments": [...]}
                    │
                    ▼
              Uniform segments: [{"start", "end", "text"}, ...]
                    │
                    ├─ (if diarize) ──► pyannote pipeline
                    │                       │
                    │                       ▼
                    │                  Speaker timeline
                    │                       │
                    │                       ▼
                    │                  assign_speakers()
                    │                  Adds "speaker" key to each segment
                    │
                    ▼
              Format to text ──► Save (CLI: .txt file, GUI: SQLite)
```

## Dependency Graph

```
Required:
  openai-whisper ─── torch, torchaudio (ML runtime)
  feedparser ─────── RSS/Atom parsing
  requests ────────── HTTP client
  python-dotenv ──── .env file loading
  certifi ─────────── SSL CA certificates
  customtkinter ──── GUI framework (GUI only)

Optional:
  faster-whisper ─── ctranslate2, tokenizers, av (CTranslate2 engine)
  pyannote.audio ─── torch (speaker diarization)
```

**Graceful degradation**: Both optional dependencies use conditional imports:
```python
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
```
UI elements (dropdowns, checkboxes) are disabled when their backing dependency is unavailable.

## Engine Comparison

| Aspect                | OpenAI Whisper              | faster-whisper                  |
|-----------------------|-----------------------------|---------------------------------|
| Backend               | PyTorch                     | CTranslate2                     |
| CPU compute type      | FP32                        | int8 quantization               |
| Relative speed (CPU)  | 1x (baseline)               | ~4x faster                      |
| Model source          | `~/.cache/whisper/`          | Hugging Face Hub                |
| `transcribe()` return | `dict` (all segments)       | `(generator, TranscriptionInfo)` |
| Progress granularity  | Per-segment (stdout capture) | Per-segment (lazy iteration)   |
| Audio duration access | `whisper.audio.load_audio()` / `SAMPLE_RATE` | `info.duration` (immediate) |
| GUI progress method   | `_WhisperProgressWriter` redirects stdout | Lazy generator iteration |
| Cancellation method   | Exception raised in stdout `write()` | Check between segment yields |
| Required dependency   | Yes (always imported)        | Optional (conditional import)   |
| GPU support           | CUDA (if available)         | CUDA, int8/float16              |

## Error Handling Strategy

| Component          | CLI behavior                   | GUI behavior                          |
|--------------------|--------------------------------|---------------------------------------|
| Feed fetch failure | `sys.exit(1)` with message     | Exception caught, shown in dialog     |
| Download failure   | Exception propagates, temp file cleaned up | Caught in worker, shown in transcript box |
| Transcription error| Exception propagates           | Caught in worker, progress resets     |
| Missing engine     | `sys.exit(1)` with install hint| Dropdown option not shown             |
| Missing HF_TOKEN   | Warning printed, diarize skipped | Checkbox works but diarize is a no-op |

All temp files are cleaned up in `finally` blocks regardless of error path.

## Security Considerations

- **No API keys for basic use**: Transcription is fully local. Only diarization requires a Hugging Face token.
- **Token storage**: `HF_TOKEN` is stored in `.env` (gitignored). Never logged or transmitted beyond Hugging Face API.
- **No remote code execution**: Audio is processed locally by Whisper/CTranslate2. No user input is passed to shell commands.
- **SSL certificate pinning**: Uses `certifi` CA bundle rather than system certificates for consistent SSL behavior across platforms.
- **Filename sanitization**: `sanitize_filename()` strips `<>:"/\|?*` and truncates to 200 chars to prevent path traversal or filesystem issues.
