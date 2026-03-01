<p align="center">
  <img src="icon.png" alt="Rangoli" width="96" />
</p>

<h1 align="center">Rangoli</h1>

<p align="center">
  <strong>Local podcast transcription with optional speaker diarization and AI analysis</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python" />
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License" />
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platform" />
</p>

---

Rangoli transcribes podcast episodes from RSS feeds using local Whisper models — no cloud API needed for transcription. It supports two engines (OpenAI Whisper and faster-whisper), optional speaker diarization via pyannote.audio, and AI-powered transcript analysis via OpenAI. Ships with both a CLI and a modern dark-themed GUI.

**Key features:**
- Two transcription engines — OpenAI Whisper (FP32) and faster-whisper (int8, ~4x faster on CPU)
- Optional speaker diarization with pyannote.audio
- AI transcript analysis via OpenAI GPT
- Dark-themed GUI with resizable sidebar, podcast artwork, paginated episodes, and real-time progress
- CLI for scripting and automation
- SQLite persistence (GUI) — podcasts, episodes, and transcripts survive across sessions
- Fully offline after initial model download (except diarization and AI analysis)

## Demo

<video src="https://github.com/user-attachments/assets/86c6baac-903b-483f-9ec0-2fbf3df9df21" controls width="100%"></video>

<details>
<summary>Can't see the video? Download it directly.</summary>

[Download demo video](docs/rangoli_demo.mp4)

</details>

## Setup

### Basic (transcription only)

```bash
git clone <repo-url> && cd rangoli
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

This installs both `openai-whisper` and `faster-whisper`. faster-whisper is optional — the tool works with just openai-whisper.

### With speaker diarization

1. Create a [Hugging Face](https://huggingface.co) account
2. Accept the [pyannote model license](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create a [token](https://huggingface.co/settings/tokens)
4. Copy `.env.example` to `.env` and add your token:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=your_token_here
```

### With AI analysis

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=sk-...
```

### macOS note

If you see duplicate Objective-C class warnings (Homebrew FFmpeg + PyAV conflict):

```bash
brew install pkg-config
pip install av --force-reinstall --no-binary av
```

## Usage

### GUI

```bash
python podcast_gui.py
```

Add podcasts via the `+` button or **File > Add Podcast** (`Cmd+N`). Right-click episodes to transcribe, analyze, or copy transcripts. The sidebar lets you pick the Whisper model, engine, and toggle diarization.

### CLI

```bash
# List episodes
python podcast_transcriber.py "https://example.com/feed.xml" --list

# Transcribe episode #3
python podcast_transcriber.py "https://example.com/feed.xml" --episode 3

# Use faster-whisper engine with a larger model
python podcast_transcriber.py "https://example.com/feed.xml" --episode 3 \
    --engine faster-whisper --model medium

# Without diarization
python podcast_transcriber.py "https://example.com/feed.xml" --episode 3 --no-diarize
```

### Configuration

All settings can be configured via `.env` (see `.env.example`) or CLI flags:

| Variable         | Default        | Description                        |
|-----------------|---------------|------------------------------------|
| `HF_TOKEN`      | —             | Hugging Face token for diarization |
| `WHISPER_MODEL`  | `base`        | Whisper model size                 |
| `WHISPER_ENGINE` | `whisper`     | `whisper` or `faster-whisper`      |
| `OUTPUT_DIR`     | `transcripts` | CLI output directory               |
| `DB_PATH`        | `podcasts.db` | SQLite database path (GUI)         |
| `OPENAI_API_KEY` | —             | OpenAI key for AI analysis         |
| `OPENAI_MODEL`   | `gpt-5.1`    | Model for AI analysis              |

### Whisper models

| Model  | Params | Speed  | Notes                  |
|--------|--------|--------|------------------------|
| tiny   | 39M    | ~32x   | Fast, lower accuracy   |
| base   | 74M    | ~16x   | Default, good balance  |
| small  | 244M   | ~6x    | Noticeably better      |
| medium | 769M   | ~2x    | High accuracy          |
| large  | 1550M  | 1x     | Best accuracy, slowest |

Models are downloaded on first use and cached at `~/.cache/whisper/`.

---

<p align="center">
  <strong>Author:</strong> Gaurav Mathur &nbsp;|&nbsp; <strong>License:</strong> MIT &nbsp;|&nbsp; <strong>2026</strong>
</p>
<p align="center">
  For architecture, design choices, and implementation details, see <a href="DESIGN.md">DESIGN.md</a>.
</p>
