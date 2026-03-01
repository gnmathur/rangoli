"""Whisper model management, progress tracking, and feature flags."""

import re
import sys
import threading
import time

import whisper

from utils import format_timestamp, estimate_remaining

# Optional faster-whisper engine
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False

# Optional diarization
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DiarizationPipeline = None
    DIARIZATION_AVAILABLE = False

# Optional OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

# ─── Model cache ─────────────────────────────────────────────────
_model_cache = {}          # (engine, model_name) -> loaded model instance
_model_cache_lock = threading.Lock()


def get_or_load_model(engine, model_name, progress_fn=None):
    """Return a cached model, loading it on first access.

    Thread-safe: concurrent calls for the same key will block until the
    first caller finishes loading.
    """
    key = (engine, model_name)
    with _model_cache_lock:
        if key in _model_cache:
            return _model_cache[key]

    # Load outside the lock (slow), then store
    if engine == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
        if progress_fn:
            progress_fn(f"Loading faster-whisper '{model_name}' (int8, cpu)...")
        mdl = WhisperModel(model_name, device="cpu", compute_type="int8")
    else:
        if progress_fn:
            progress_fn(f"Loading whisper '{model_name}'...")
        mdl = whisper.load_model(model_name)

    with _model_cache_lock:
        _model_cache.setdefault(key, mdl)  # first writer wins
        return _model_cache[key]


class TranscriptionCancelled(Exception):
    """Raised when the user stops a transcription in progress."""


# Regex to extract end timestamp from whisper verbose output:
#   [00:05.123 --> 00:12.456] text   or   [01:05:32.123 --> 01:05:40.456] text
WHISPER_TS_RE = re.compile(r'-->\s*(?:(\d+):)?(\d+):(\d+\.\d+)\]')


class WhisperProgressWriter:
    """Captures whisper verbose stdout to drive GUI progress updates.

    Replaces sys.stdout during model.transcribe(verbose=True) so that each
    printed segment line is parsed for its end timestamp, giving the same
    per-segment progress that faster-whisper provides natively.
    """

    def __init__(self, audio_duration, stage_t0, update_fn, model_name, cancel_event):
        self._audio_duration = audio_duration
        self._stage_t0 = stage_t0
        self._update_fn = update_fn
        self._model_name = model_name
        self._cancel_event = cancel_event
        self._duration_str = format_timestamp(audio_duration)
        self.segment_count = 0
        self._buf = ""

    def write(self, text):
        if self._cancel_event.is_set():
            raise TranscriptionCancelled()
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._parse_line(line)
        return len(text)

    def _parse_line(self, line):
        m = WHISPER_TS_RE.search(line)
        if not m:
            return
        self.segment_count += 1
        h = int(m.group(1)) if m.group(1) else 0
        mins = int(m.group(2))
        secs = float(m.group(3))
        end_time = h * 3600 + mins * 60 + secs

        seg_pct = min(end_time / self._audio_duration, 1.0) if self._audio_duration else 0
        progress = 0.35 + seg_pct * 0.50
        elapsed = time.monotonic() - self._stage_t0
        eta = estimate_remaining(elapsed, seg_pct)
        eta_str = f" — ~{eta} remaining" if eta else ""
        self._update_fn(
            progress, "Stage 3/4: Transcribing",
            f"whisper '{self._model_name}' — "
            f"{self.segment_count} segments | "
            f"{format_timestamp(end_time)} / {self._duration_str}"
            f"{eta_str}")

    def flush(self):
        pass

    # Forward attribute lookups (encoding, errors, etc.) to real stdout
    def __getattr__(self, name):
        return getattr(sys.__stdout__, name)
