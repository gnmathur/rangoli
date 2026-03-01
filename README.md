# Rangoli — Podcast Transcriber

**Version 1.0.0** | **Author: Gaurav Mathur** | **License: MIT** | **2026**

Transcribes podcast episodes from RSS feeds using local OpenAI Whisper or faster-whisper, with optional speaker diarization via pyannote.audio. Includes both a CLI and a modern dark-themed GUI.

## Design Choices

- **Local transcription**: Uses Whisper running locally - no API keys needed for basic transcription, fully offline capable after model download
- **Dual engine support**: Supports both OpenAI Whisper and faster-whisper (CTranslate2). faster-whisper provides ~4x faster CPU inference via int8 quantization with comparable accuracy
- **RSS-first approach**: Takes a standard podcast RSS feed URL as input, making it compatible with virtually any podcast
- **Optional diarization**: Speaker diarization via pyannote.audio is opt-in; the tool works without it if you just need a plain transcript
- **Temporary audio**: Downloaded episode audio is stored in a temp file and deleted after transcription to avoid disk bloat
- **Segment-level output**: Preserves Whisper's timestamp segments for navigable transcripts
- **SQLite persistence (GUI)**: The GUI uses SQLite to persist podcasts, episodes, and generated transcripts across sessions. No external database needed
- **CustomTkinter GUI**: Modern dark-mode UI using CustomTkinter for a polished Material-style look without heavy Qt/Electron dependencies
- **Stage-based progress reporting**: The GUI shows a 4-stage progress pipeline (download, model loading, transcription, diarization) with a detail line showing per-segment metrics. Both engines provide identical real-time per-segment progress with segment count, audio time processed, and live ETA
- **Model cached at `~/.cache/whisper/`**: Whisper models are downloaded once and cached in the user's home directory for reuse
- **Duration normalization**: Episode durations from RSS feeds (which vary between raw seconds, MM:SS, and HH:MM:SS) are normalized to a compact human-readable format (e.g. "1h 23m 45s")
- **Publish date column**: Episode table shows formatted publish dates (e.g. "Feb 24, 2026") parsed from RFC 2822 feed dates
- **Resizable sidebar**: The sidebar uses a PanedWindow so the user can drag to resize the sidebar and main area
- **Auto-refresh on startup**: All podcast feeds are refreshed in the background when the app launches
- **Podcast info panel**: Right-click a podcast to open a 3rd column showing artwork, name, and description with copy-to-clipboard support
- **macOS menu bar integration**: On macOS, the application sets its process name to "Rangoli" via the Foundation framework and provides a native menu bar with File menu and About dialog

## Design Tradeoffs

- **Whisper model size vs. speed**: Defaults to `base` model (reasonable accuracy, moderate speed). Larger models (`medium`, `large`) give better accuracy but are significantly slower and need more RAM/VRAM
- **faster-whisper tradeoffs**: Uses int8 quantization on CPU for speed. Accuracy is comparable to the original Whisper but not identical — int8 quantization may produce slightly different results. faster-whisper is an optional dependency; the tool falls back to standard Whisper if not installed
- **pyannote.audio requires Hugging Face token**: Speaker diarization requires accepting model license terms on Hugging Face - adds friction but pyannote is the most accurate open-source diarization available
- **Speaker assignment by overlap**: Speakers are assigned to Whisper segments by finding the diarization segment with the greatest time overlap. This works well but can mis-assign short segments at speaker boundaries
- **No audio caching**: Episodes are re-downloaded on each run. This keeps things simple but means re-transcribing requires re-downloading
- **Threaded transcription in GUI**: Transcription runs in a background thread to keep the GUI responsive, with progress updates pushed to the UI via `after()` callbacks. Cooperative cancellation via `threading.Event` allows stopping at every checkpoint (download chunks, segment iterations, stage boundaries)
- **Uniform progress across engines**: Both engines show identical per-segment progress. faster-whisper yields segments via a lazy generator; OpenAI Whisper's verbose output is captured via stdout redirect (`_WhisperProgressWriter`) to parse segment timestamps as they're printed, achieving the same effect
- **SQLite for GUI only**: The CLI still writes plain `.txt` files for simplicity; SQLite is used only by the GUI. This keeps the CLI zero-dependency on a database
- **macOS process name via CoreFoundation ctypes**: Setting the menu bar app name to "Rangoli" uses CoreFoundation's C API via `ctypes` to modify `CFBundleName` before Tk initializes. No external dependencies required — uses only macOS system libraries

## Setup

### Basic (transcription only)

```bash
pip install -r requirements.txt
```

This installs both `openai-whisper` and `faster-whisper`. If you only want one engine, install dependencies manually. faster-whisper is optional — the tool works with just openai-whisper installed.

### With speaker diarization

1. Create a [Hugging Face](https://huggingface.co) account
2. Accept the pyannote model license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create a token at https://huggingface.co/settings/tokens
4. Copy `.env.example` to `.env` and add your token:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=your_token_here
```

### macOS note

If you have FFmpeg installed via Homebrew and also install `faster-whisper` (which pulls in PyAV with bundled FFmpeg), you may see duplicate Objective-C class warnings. Fix by rebuilding PyAV against your system FFmpeg:

```bash
brew install pkg-config
pip install av --force-reinstall --no-binary av
```

## Usage

### GUI Mode

```bash
python podcast_gui.py
```

The GUI provides:
- **Menu bar**: Native macOS menu bar showing "Rangoli" with File menu (Add Podcast, Remove Podcast) and About dialog showing version, author, and license info
- **Add podcasts**: Click the `+` button or use File > Add Podcast (Cmd+N) to enter an RSS feed URL
- **Browse episodes**: Select a podcast to see a paginated list of episodes (15 per page) sorted by publish date (newest first) with formatted dates and durations in compact format (e.g. "1h 23m 45s")
- **Resizable sidebar**: Drag the sidebar edge to resize the sidebar and main area
- **Transcript panel**: Selecting an episode or completing transcription opens a 3rd column panel showing the transcript text. Use the Copy button to copy to clipboard, Close to hide the panel
- **Podcast info**: Right-click a podcast in the sidebar to open a modal dialog showing artwork, name, and description
- **View menu**: Switch between Dark Mode and Light Mode via the View menu
- **Auto-refresh**: On startup, all podcast feeds are refreshed in the background to pick up new episodes
- **Select Whisper model**: Choose from tiny/base/small/medium/large in the sidebar dropdown
- **Select engine**: Switch between `whisper` and `faster-whisper` engines via sidebar dropdown (faster-whisper option disabled if not installed)
- **Transcribe**: Select an episode and click "Transcribe" - a 4-stage progress bar tracks download, model loading, transcription, and diarization. Both engines show identical per-segment progress with segment count, audio time processed, and a live time-remaining estimate. Total wall-clock time is shown on completion
- **Stop transcription**: The "Transcribe" button becomes a red "Stop" button during transcription. Clicking it cancels the current job — both engines stop within one segment. Temp files are cleaned up regardless
- **View transcripts**: Completed transcripts are shown in a scrollable text viewer and persisted in SQLite
- **Diarization toggle**: Enable/disable speaker diarization via checkbox (requires pyannote.audio + HF_TOKEN)

### CLI Mode

```bash
# List episodes from a podcast feed
python podcast_transcriber.py "https://example.com/feed.xml" --list

# Transcribe episode #3
python podcast_transcriber.py "https://example.com/feed.xml" --episode 3

# Transcribe without speaker diarization
python podcast_transcriber.py "https://example.com/feed.xml" --episode 3 --no-diarize

# Use a larger Whisper model for better accuracy
python podcast_transcriber.py "https://example.com/feed.xml" --episode 3 --model medium

# Custom output directory
python podcast_transcriber.py "https://example.com/feed.xml" --episode 3 -o ./my_transcripts

# Use faster-whisper engine for ~4x faster CPU transcription
python podcast_transcriber.py "https://example.com/feed.xml" --episode 3 --engine faster-whisper
```

### Whisper Model Sizes

| Model  | Parameters | Relative Speed | Notes                    |
|--------|-----------|----------------|--------------------------|
| tiny   | 39M       | ~32x           | Fast, lower accuracy     |
| base   | 74M       | ~16x           | Default, good balance    |
| small  | 244M      | ~6x            | Noticeably better        |
| medium | 769M      | ~2x            | High accuracy            |
| large  | 1550M     | 1x             | Best accuracy, slowest   |

Models are downloaded on first use and cached at `~/.cache/whisper/`.

## Configuration

Settings can be configured via `.env` file or command-line flags (flags take precedence):

| Variable        | Default        | Description                          |
|----------------|---------------|--------------------------------------|
| `HF_TOKEN`     | (none)        | Hugging Face token for diarization   |
| `WHISPER_MODEL` | `base`       | Default Whisper model size           |
| `WHISPER_ENGINE`| `whisper`    | Transcription engine (`whisper` or `faster-whisper`) |
| `OUTPUT_DIR`   | `transcripts` | Default output directory (CLI)       |
| `DB_PATH`      | `podcasts.db` | SQLite database path (GUI)           |

## Output Format

**GUI**: Transcripts are stored in SQLite and displayed in the built-in viewer.

**CLI**: Transcripts are saved as `.txt` files in the output directory:

**With diarization:**
```
Podcast: My Podcast
Episode: Episode Title
Transcribed: 2026-02-28 10:30:00
============================================================

[Speaker 1] (00:00:05)
Welcome to the show. Today we're talking about...

[Speaker 2] (00:00:12)
Thanks for having me. I'm excited to discuss...
```

**Without diarization:**
```
Podcast: My Podcast
Episode: Episode Title
Transcribed: 2026-02-28 10:30:00
============================================================

[00:00:05] Welcome to the show. Today we're talking about...
[00:00:12] Thanks for having me. I'm excited to discuss...
```

## Architecture

```
rangoli/
  podcast_gui.py          # GUI application (CustomTkinter)
  podcast_transcriber.py  # CLI application
  database.py             # SQLite database layer
  icon.png                # Application icon (rangoli)
  LICENSE                 # MIT License
  requirements.txt
  .env.example
  DESIGN.md               # Detailed system design document
  transcripts/            # CLI output directory
  podcasts.db             # SQLite database (created on first GUI run)
```

## Implementation Notes

- RSS parsing uses `feedparser` which handles various feed formats (RSS 2.0, Atom, etc.)
- Audio enclosure detection checks `<enclosure>` tags, `<media:content>`, and link extensions as fallbacks
- OpenAI Whisper transcription explicitly uses `fp16=False` since FP16 is not supported on CPU - this avoids the runtime warning and uses FP32 instead
- faster-whisper uses CTranslate2 with `compute_type="int8"` on CPU for optimized inference
- Whisper outputs timestamped segments; diarization produces a separate speaker timeline. These are merged by computing time overlap between each Whisper segment and diarization turns
- Speaker labels are mapped to friendly names (`Speaker 1`, `Speaker 2`, ...) in order of first appearance
- Filenames are sanitized to remove filesystem-unsafe characters
- The GUI uses `threading` for non-blocking transcription; all UI updates go through `after()` to stay on the main thread
- SQLite uses `PRAGMA foreign_keys = ON` with cascading deletes so removing a podcast cleans up its episodes and transcripts
- The database schema stores transcripts with metadata (model used, diarization flag, timestamp) for reproducibility
- Optional dependencies (`faster-whisper`, `pyannote.audio`) are wrapped in try/except imports with availability flags, so the app degrades gracefully when they are not installed
- Episode durations from RSS feeds are normalized from various formats (seconds, MM:SS, HH:MM:SS) to a compact human-readable format (e.g. "1h 23m 45s")
- macOS menu bar name is set via CoreFoundation `ctypes` (no external deps) before tkinter import
- Application icon (rangoli) is set via `iconphoto()` for the window and dock icon
