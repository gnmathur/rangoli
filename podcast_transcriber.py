#!/usr/bin/env python3
"""
Podcast Transcriber

Transcribes podcast episodes from RSS feeds using local Whisper
with optional speaker diarization via pyannote.audio.
"""

import argparse
import os
import re
import ssl
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import certifi
import feedparser
import requests
import whisper
from dotenv import load_dotenv

# Fix macOS Python SSL cert issue (urllib uses system certs which may be missing)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

load_dotenv()

# Optional faster-whisper engine
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Optional diarization
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False


def fetch_feed(rss_url):
    """Parse RSS feed and return podcast metadata with episodes."""
    # Use requests to fetch feed content (handles SSL properly via certifi)
    try:
        response = requests.get(rss_url, timeout=30)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except requests.RequestException as e:
        print(f"Error: Could not fetch feed at {rss_url}: {e}")
        sys.exit(1)

    if feed.bozo and not feed.entries:
        print(f"Error: Could not parse feed at {rss_url}")
        sys.exit(1)

    episodes = []
    for entry in feed.entries:
        audio_url = None

        # Check enclosures first (standard RSS podcast format)
        for enclosure in entry.get("enclosures", []):
            if enclosure.get("type", "").startswith("audio/"):
                audio_url = enclosure.get("href")
                break

        # Fallback to media:content
        if not audio_url:
            for media in entry.get("media_content", []):
                if media.get("type", "").startswith("audio/"):
                    audio_url = media.get("url")
                    break

        # Fallback: check for links with audio extensions
        if not audio_url:
            for link in entry.get("links", []):
                href = link.get("href", "")
                if any(href.lower().endswith(ext) for ext in [".mp3", ".m4a", ".wav", ".ogg", ".aac"]):
                    audio_url = href
                    break

        if audio_url:
            episodes.append({
                "title": entry.get("title", "Untitled"),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", "")[:200] if entry.get("summary") else "",
                "audio_url": audio_url,
                "duration": entry.get("itunes_duration", ""),
            })

    return {
        "title": feed.feed.get("title", "Unknown Podcast"),
        "author": feed.feed.get("author", feed.feed.get("itunes_author", "")),
        "episodes": episodes,
    }


def list_episodes(podcast):
    """Print a numbered list of episodes."""
    print(f"\n  {podcast['title']}")
    if podcast["author"]:
        print(f"  by {podcast['author']}")
    print(f"  {len(podcast['episodes'])} episodes with audio\n")

    for i, ep in enumerate(podcast["episodes"], 1):
        duration = f" ({ep['duration']})" if ep["duration"] else ""
        published = f"  {ep['published']}" if ep["published"] else ""
        print(f"  {i:3d}. {ep['title']}{duration}")
        if published:
            print(f"       {published}")
    print()


def download_audio(url, output_path):
    """Download audio file with progress indication."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    downloaded = 0
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Downloading: {pct:.1f}%", end="", flush=True)
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\r  Downloading: {mb:.1f} MB", end="", flush=True)
    print()


def transcribe_audio(audio_path, model_name="base", engine="whisper"):
    """Transcribe audio using Whisper or faster-whisper."""
    if engine == "faster-whisper":
        if not FASTER_WHISPER_AVAILABLE:
            print("Error: faster-whisper is not installed. Install with: pip install faster-whisper")
            sys.exit(1)
        print(f"  Loading faster-whisper model '{model_name}' (int8)...")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print("  Transcribing (this may take a while)...")
        fw_segments, info = model.transcribe(str(audio_path))
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in fw_segments]
        print(f"  Transcription complete: {len(segments)} segments")
        return {"segments": segments}
    else:
        print(f"  Loading Whisper model '{model_name}'...")
        model = whisper.load_model(model_name)
        print("  Transcribing (this may take a while)...")
        result = model.transcribe(str(audio_path), verbose=False, fp16=False)
        print(f"  Transcription complete: {len(result['segments'])} segments")
        return result


def diarize_audio(audio_path):
    """Perform speaker diarization using pyannote.audio."""
    if not DIARIZATION_AVAILABLE:
        print("  Warning: pyannote.audio not installed. Skipping diarization.")
        print("  Install with: pip install pyannote.audio")
        return None

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "your_huggingface_token_here":
        print("  Warning: HF_TOKEN not set in .env. Skipping diarization.")
        print("  See .env.example for setup instructions.")
        return None

    print("  Running speaker diarization...")
    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarization = pipeline(str(audio_path))
    return diarization


def assign_speakers(segments, diarization):
    """Assign speaker labels to Whisper segments based on diarization overlap."""
    if diarization is None:
        return segments

    # Build speaker timeline from diarization
    speaker_timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_timeline.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # Map speaker IDs to friendlier names (Speaker 1, Speaker 2, ...)
    speaker_names = {}
    name_counter = 1

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Find speaker with greatest overlap
        best_speaker = None
        best_overlap = 0

        for sp in speaker_timeline:
            overlap_start = max(seg_start, sp["start"])
            overlap_end = min(seg_end, sp["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp["speaker"]

        if best_speaker and best_speaker not in speaker_names:
            speaker_names[best_speaker] = f"Speaker {name_counter}"
            name_counter += 1

        seg["speaker"] = speaker_names.get(best_speaker, "Unknown") if best_speaker else "Unknown"

    num_speakers = len(speaker_names)
    print(f"  Identified {num_speakers} speaker(s)")
    return segments


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(segments, podcast_title, episode_title, has_speakers=False):
    """Format transcript segments into readable text."""
    lines = [
        f"Podcast: {podcast_title}",
        f"Episode: {episode_title}",
        f"Transcribed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    if has_speakers:
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            timestamp = format_timestamp(seg["start"])
            text = seg["text"].strip()

            if speaker != current_speaker:
                current_speaker = speaker
                lines.append(f"\n[{speaker}] ({timestamp})")
            lines.append(text)
    else:
        for seg in segments:
            timestamp = format_timestamp(seg["start"])
            text = seg["text"].strip()
            lines.append(f"[{timestamp}] {text}")

    return "\n".join(lines)


def sanitize_filename(name):
    """Make a string safe for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name)
    return name[:200].strip()


def transcribe_episode(podcast, episode_index, model_name, output_dir, diarize, engine="whisper"):
    """Download and transcribe a single episode."""
    episodes = podcast["episodes"]
    if episode_index < 1 or episode_index > len(episodes):
        print(f"Error: Episode number must be between 1 and {len(episodes)}")
        sys.exit(1)

    episode = episodes[episode_index - 1]
    print(f"\n  Podcast:  {podcast['title']}")
    print(f"  Episode:  {episode['title']}")
    if episode["duration"]:
        print(f"  Duration: {episode['duration']}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download audio to temp file
    audio_url = episode["audio_url"]
    extension = Path(audio_url.split("?")[0]).suffix or ".mp3"

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        download_audio(audio_url, tmp_path)

        # Transcribe
        result = transcribe_audio(tmp_path, model_name, engine)
        segments = result["segments"]

        # Diarize if requested
        has_speakers = False
        if diarize:
            diarization = diarize_audio(tmp_path)
            if diarization is not None:
                segments = assign_speakers(segments, diarization)
                has_speakers = True

        # Format and save
        transcript = format_transcript(
            segments, podcast["title"], episode["title"], has_speakers
        )

        podcast_name = sanitize_filename(podcast["title"])
        episode_name = sanitize_filename(episode["title"])
        filename = f"{podcast_name} - {episode_name}.txt"
        filepath = output_path / filename

        filepath.write_text(transcript, encoding="utf-8")
        print(f"\n  Saved: {filepath}")

    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe podcast episodes from RSS feeds using Whisper"
    )
    parser.add_argument("rss_url", help="Podcast RSS feed URL")
    parser.add_argument(
        "--list", action="store_true", help="List episodes and exit"
    )
    parser.add_argument(
        "--episode", "-e", type=int,
        help="Episode number to transcribe (from --list output)"
    )
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("WHISPER_MODEL", "base"),
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.getenv("OUTPUT_DIR", "transcripts"),
        help="Output directory (default: transcripts)",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip speaker diarization",
    )
    parser.add_argument(
        "--engine",
        default=os.getenv("WHISPER_ENGINE", "whisper"),
        choices=["whisper", "faster-whisper"],
        help="Transcription engine (default: whisper)",
    )

    args = parser.parse_args()

    print(f"  Fetching feed: {args.rss_url}")
    podcast = fetch_feed(args.rss_url)

    if not podcast["episodes"]:
        print("No episodes with audio found in this feed.")
        sys.exit(1)

    if args.list or args.episode is None:
        list_episodes(podcast)
        if args.episode is None and not args.list:
            print("Use --episode N to transcribe an episode, or --list to list them.")
        return

    transcribe_episode(
        podcast=podcast,
        episode_index=args.episode,
        model_name=args.model,
        output_dir=args.output,
        diarize=not args.no_diarize,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
