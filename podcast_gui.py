#!/usr/bin/env python3
"""
Rangoli — Podcast Transcriber

A CustomTkinter-based GUI for subscribing to podcasts, browsing episodes,
transcribing with Whisper, and viewing transcripts.

Author:  Gaurav Mathur
License: MIT
"""

import email.utils
import io
import os
import re
import ssl
import sys
import tempfile
import threading
import time
from pathlib import Path

APP_NAME = "Rangoli"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Gaurav Mathur"
APP_YEAR = "2026"

# Set macOS menu bar app name via CoreFoundation ctypes — MUST run before tkinter import
if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util
        _cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        _cf.CFBundleGetMainBundle.restype = ctypes.c_void_p
        _cf.CFBundleGetInfoDictionary.restype = ctypes.c_void_p
        _cf.CFBundleGetInfoDictionary.argtypes = [ctypes.c_void_p]
        _cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        _cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        _cf.CFDictionarySetValue.restype = None
        _cf.CFDictionarySetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        _cf.CFRelease.argtypes = [ctypes.c_void_p]
        _kCFStringEncodingUTF8 = 0x08000100
        _bundle = _cf.CFBundleGetMainBundle()
        _info = _cf.CFBundleGetInfoDictionary(_bundle)
        _key = _cf.CFStringCreateWithCString(None, b"CFBundleName", _kCFStringEncodingUTF8)
        _val = _cf.CFStringCreateWithCString(None, APP_NAME.encode(), _kCFStringEncodingUTF8)
        _cf.CFDictionarySetValue(_info, _key, _val)
        _cf.CFRelease(_key)
        _cf.CFRelease(_val)
        del _cf, _bundle, _info, _key, _val, _kCFStringEncodingUTF8
    except Exception:
        pass

import tkinter as tk  # noqa: E402 — must be after CFBundleName override

import certifi  # noqa: E402
import customtkinter as ctk  # noqa: E402
import feedparser  # noqa: E402
import requests  # noqa: E402
import whisper  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import database as db  # noqa: E402

# SSL fix for macOS
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

# ─── Model cache ─────────────────────────────────────────────────
_model_cache = {}          # (engine, model_name) → loaded model instance
_model_cache_lock = threading.Lock()


def _get_or_load_model(engine, model_name, progress_fn=None):
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


class _TranscriptionCancelled(Exception):
    """Raised when the user stops a transcription in progress."""


# Regex to extract end timestamp from whisper verbose output:
#   [00:05.123 --> 00:12.456] text   or   [01:05:32.123 --> 01:05:40.456] text
_WHISPER_TS_RE = re.compile(r'-->\s*(?:(\d+):)?(\d+):(\d+\.\d+)\]')


class _WhisperProgressWriter:
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
            raise _TranscriptionCancelled()
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._parse_line(line)
        return len(text)

    def _parse_line(self, line):
        m = _WHISPER_TS_RE.search(line)
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
        eta = _estimate_remaining(elapsed, seg_pct)
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


SIDEBAR_ICON_SIZE = 24


def _make_square_icon(pil_image, size=SIDEBAR_ICON_SIZE):
    """Resize a PIL image to a uniform square CTkImage using center-crop."""
    img = pil_image.convert("RGBA")
    img = ImageOps.fit(img, (size, size), method=Image.LANCZOS, centering=(0.5, 0.5))
    return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))


WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
WHISPER_ENGINES = ["whisper", "faster-whisper"]
EPISODE_ROW_HEIGHT = 52  # title + blurb + padding
EPISODE_OVERHEAD = 110  # top bar + progress area + padding
MIN_EPISODES_PER_PAGE = 5
ICON_PATH = Path(__file__).parent / "icon.png"

# Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def fetch_feed(rss_url):
    """Parse RSS feed and return podcast metadata with episodes."""
    response = requests.get(rss_url, timeout=30)
    response.raise_for_status()
    feed = feedparser.parse(response.content)

    if feed.bozo and not feed.entries:
        raise ValueError(f"Could not parse feed at {rss_url}")

    episodes = []
    for entry in feed.entries:
        audio_url = None
        for enclosure in entry.get("enclosures", []):
            if enclosure.get("type", "").startswith("audio/"):
                audio_url = enclosure.get("href")
                break
        if not audio_url:
            for media in entry.get("media_content", []):
                if media.get("type", "").startswith("audio/"):
                    audio_url = media.get("url")
                    break
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

    description = feed.feed.get("description") or feed.feed.get("subtitle", "")
    image_url = ""
    if feed.feed.get("image", {}).get("href"):
        image_url = feed.feed["image"]["href"]
    elif feed.feed.get("itunes_image", {}).get("href"):
        image_url = feed.feed["itunes_image"]["href"]

    return {
        "title": feed.feed.get("title", "Unknown Podcast"),
        "author": feed.feed.get("author", feed.feed.get("itunes_author", "")),
        "description": description,
        "image_url": image_url,
        "episodes": episodes,
    }


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_duration(seconds):
    """Format seconds as human-readable duration (e.g. '2m 15s', '45s')."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _estimate_remaining(elapsed, fraction_done):
    """Estimate time remaining given elapsed time and fraction completed (0-1).

    Returns formatted string, or None if not enough data yet.
    """
    if fraction_done <= 0 or elapsed < 2:
        return None
    total_est = elapsed / fraction_done
    remaining = total_est - elapsed
    if remaining < 1:
        return None
    return _format_duration(remaining)


_HTML_TAG_RE = re.compile(r'<[^>]+>')
_HTML_ENTITY_RE = re.compile(r'&\w+;|&#\d+;')


def _strip_html(text):
    """Remove HTML tags and collapse whitespace."""
    text = _HTML_TAG_RE.sub(' ', text)
    text = _HTML_ENTITY_RE.sub(' ', text)
    return ' '.join(text.split())


def _normalize_duration(raw):
    """Normalize iTunes duration to compact format (e.g. '1h 23m 45s').

    Handles: seconds ("3600"), "MM:SS" ("45:30"), "HH:MM:SS" ("1:23:45").
    """
    if not raw:
        return ""
    raw = raw.strip()

    h = m = s = 0
    # Pure seconds (e.g. "3600")
    try:
        total = int(raw)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
    except ValueError:
        # MM:SS or HH:MM:SS
        parts = raw.split(":")
        if len(parts) == 2:
            try:
                m, s = int(parts[0]), int(parts[1])
                h, m = divmod(m, 60)
            except ValueError:
                return raw
        elif len(parts) == 3:
            try:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                return raw
        else:
            return raw

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_publish_date(raw):
    """Format an RFC 2822 date string as 'Feb 24, 2026'. Falls back to raw string."""
    if not raw:
        return ""
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        return dt.strftime("%b %d, %Y")
    except Exception:
        return raw


class AddPodcastDialog(ctk.CTkToplevel):
    """Modal dialog to add a new podcast by RSS URL."""

    def __init__(self, parent, on_add_callback):
        super().__init__(parent)
        self.on_add_callback = on_add_callback
        self.title(f"{APP_NAME} — Add Podcast")
        self.geometry("500x180")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - 500) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - 180) // 2
        self.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(frame, text="Podcast RSS Feed URL",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")

        self.url_entry = ctk.CTkEntry(frame, placeholder_text="https://feeds.example.com/podcast.xml",
                                       width=460)
        self.url_entry.pack(pady=(8, 12), fill="x")
        self.url_entry.bind("<Return>", lambda e: self._on_add())

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x")

        self.status_label = ctk.CTkLabel(btn_frame, text="", text_color="gray")
        self.status_label.pack(side="left")

        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="gray",
                       hover_color="#555", command=self.destroy).pack(side="right", padx=(8, 0))
        self.add_btn = ctk.CTkButton(btn_frame, text="Add", width=80, command=self._on_add)
        self.add_btn.pack(side="right")

        self.url_entry.focus()

    def _on_add(self):
        url = self.url_entry.get().strip()
        if not url:
            return
        self.add_btn.configure(state="disabled")
        self.status_label.configure(text="Fetching feed...", text_color="#4a9eff")
        threading.Thread(target=self._fetch_and_add, args=(url,), daemon=True).start()

    def _fetch_and_add(self, url):
        try:
            podcast_data = fetch_feed(url)
            podcast_id = db.add_podcast(
                url, podcast_data["title"], podcast_data["author"], podcast_data["episodes"],
                description=podcast_data.get("description", ""),
                image_url=podcast_data.get("image_url", ""),
            )
            if podcast_id is None:
                self.after(0, lambda: self.status_label.configure(
                    text="Podcast already exists", text_color="#ff6b6b"))
                self.after(0, lambda: self.add_btn.configure(state="normal"))
                return
            self.after(0, lambda: self._done(podcast_id))
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(
                text=f"Error: {e}", text_color="#ff6b6b"))
            self.after(0, lambda: self.add_btn.configure(state="normal"))

    def _done(self, podcast_id):
        self.on_add_callback()
        self.destroy()


class PodcastApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        db.init_db()

        self.title(APP_NAME)
        self.geometry("1200x750")
        self.minsize(900, 550)

        # Application icon (tk.PhotoImage required by iconphoto API)
        if ICON_PATH.exists():
            self._icon = tk.PhotoImage(file=str(ICON_PATH))
            self.iconphoto(True, self._icon)

        self.selected_podcast_id = None
        self.episode_page = 0
        self.episode_total = 0
        self._transcribing = False
        self._cancel_event = threading.Event()

        self._transcript_panel = None
        self._transcript_panel_visible = False
        self._podcast_icon_cache = {}  # podcast_id → CTkImage (24x24)

        self._last_eps_per_page = MIN_EPISODES_PER_PAGE

        self._build_menubar()
        self._build_ui()
        self._load_podcasts()

        # Reload episode list when window resizes so page fills available height
        self._resize_after_id = None
        self.bind("<Configure>", self._on_resize)

        # Auto-refresh feeds in background
        threading.Thread(target=self._refresh_all_podcasts, daemon=True).start()

        # Pre-load base and small whisper models in background
        self._preload_models()

    # ─── Menu bar ─────────────────────────────────────────────────────

    def _build_menubar(self):
        menubar = tk.Menu(self)

        # macOS application menu (apple menu)
        if sys.platform == "darwin":
            app_menu = tk.Menu(menubar, name="apple")
            menubar.add_cascade(menu=app_menu)
            app_menu.add_command(label=f"About {APP_NAME}", command=self._show_about)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Add Podcast...", command=self._open_add_dialog,
                              accelerator="Command+N" if sys.platform == "darwin" else "Ctrl+N")
        file_menu.add_command(label="Remove Podcast", command=self._delete_podcast)
        if sys.platform != "darwin":
            file_menu.add_separator()
            file_menu.add_command(label="Quit", command=self.destroy,
                                  accelerator="Ctrl+Q")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Dark Mode", command=lambda: self._set_appearance("dark"))
        view_menu.add_command(label="Light Mode", command=lambda: self._set_appearance("light"))

        # Help menu (non-macOS; on macOS About is in apple menu)
        if sys.platform != "darwin":
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label=f"About {APP_NAME}", command=self._show_about)

        self.config(menu=menubar)

        # Keyboard shortcuts
        mod = "Command" if sys.platform == "darwin" else "Control"
        self.bind(f"<{mod}-n>", lambda e: self._open_add_dialog())
        if sys.platform != "darwin":
            self.bind("<Control-q>", lambda e: self.destroy())

    def _show_about(self):
        about = ctk.CTkToplevel(self)
        about.title(f"About {APP_NAME}")
        about.geometry("340x280")
        about.resizable(False, False)
        about.transient(self)
        about.grab_set()

        about.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - 340) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 280) // 2
        about.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(about, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=24, pady=20)

        # Icon (use CTkImage for HighDPI support)
        if ICON_PATH.exists():
            icon_img = ctk.CTkImage(light_image=Image.open(ICON_PATH), size=(64, 64))
            icon_label = ctk.CTkLabel(frame, text="", image=icon_img)
            icon_label.pack(pady=(0, 8))

        ctk.CTkLabel(frame, text=APP_NAME,
                     font=ctk.CTkFont(size=22, weight="bold")).pack()
        ctk.CTkLabel(frame, text=f"Version {APP_VERSION}",
                     font=ctk.CTkFont(size=13), text_color="gray").pack(pady=(2, 8))
        ctk.CTkLabel(frame, text="Podcast Transcriber",
                     font=ctk.CTkFont(size=13)).pack()
        ctk.CTkLabel(frame, text=f"{APP_AUTHOR}",
                     font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(4, 2))
        ctk.CTkLabel(frame, text=f"Copyright \u00a9 {APP_YEAR} {APP_AUTHOR}",
                     font=ctk.CTkFont(size=11), text_color="gray").pack()
        ctk.CTkLabel(frame, text="MIT License",
                     font=ctk.CTkFont(size=11), text_color="gray").pack()

        ctk.CTkButton(frame, text="OK", width=80, command=about.destroy).pack(pady=(12, 0))

    def _set_appearance(self, mode):
        """Switch between dark and light mode."""
        ctk.set_appearance_mode(mode)
        sash_color = "#333333" if mode == "dark" else "#cccccc"
        bg_color = "#333333" if mode == "dark" else "#ebebeb"
        self._paned.configure(bg=bg_color, sashpad=0)

    # ─── UI Construction ─────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Horizontal PanedWindow: sidebar | main area (| info panel)
        self._paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5,
                                      bg="#333333", bd=0)
        self._paned.grid(row=0, column=0, sticky="nsew")

        # Left sidebar
        self._sidebar = self._build_sidebar()
        self._paned.add(self._sidebar, minsize=220, width=320)

        # Right main area
        self._main_frame = self._build_main_area()
        self._paned.add(self._main_frame, minsize=500)

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self._paned, corner_radius=0)
        sidebar.grid_rowconfigure(3, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        # Header with icon and app name
        header = ctk.CTkFrame(sidebar, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))

        if ICON_PATH.exists():
            header_icon = ctk.CTkImage(light_image=Image.open(ICON_PATH), size=(20, 20))
            ctk.CTkLabel(header, text="", image=header_icon).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(header, text=APP_NAME,
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        ctk.CTkButton(header, text="+", width=32, height=32,
                       font=ctk.CTkFont(size=18),
                       command=self._open_add_dialog).pack(side="right")

        # Settings: Model, Engine, Diarize — grid-aligned
        settings_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        settings_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))
        settings_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings_frame, text="Model:", font=ctk.CTkFont(size=12)
                     ).grid(row=0, column=0, sticky="w", pady=2)
        default_model = os.getenv("WHISPER_MODEL", "base")
        self.model_var = ctk.StringVar(value=default_model)
        ctk.CTkOptionMenu(settings_frame, variable=self.model_var, values=WHISPER_MODELS,
                           width=140, height=28).grid(row=0, column=1, sticky="w", padx=(6, 0), pady=2)

        ctk.CTkLabel(settings_frame, text="Engine:", font=ctk.CTkFont(size=12)
                     ).grid(row=1, column=0, sticky="w", pady=2)
        default_engine = os.getenv("WHISPER_ENGINE", "whisper")
        if not FASTER_WHISPER_AVAILABLE and default_engine == "faster-whisper":
            default_engine = "whisper"
        self.engine_var = ctk.StringVar(value=default_engine)
        engine_values = WHISPER_ENGINES if FASTER_WHISPER_AVAILABLE else ["whisper"]
        ctk.CTkOptionMenu(settings_frame, variable=self.engine_var, values=engine_values,
                           width=140, height=28).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=2)

        self.diarize_var = ctk.BooleanVar(value=DIARIZATION_AVAILABLE)
        diarize_cb = ctk.CTkCheckBox(settings_frame, text="Diarize", variable=self.diarize_var,
                                      height=28, checkbox_width=18, checkbox_height=18)
        diarize_cb.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        if not DIARIZATION_AVAILABLE:
            diarize_cb.configure(state="disabled")

        # Podcast list (scrollable)
        self.podcast_list_frame = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        self.podcast_list_frame.grid(row=3, column=0, sticky="nsew", padx=6, pady=4)

        # Delete button at bottom
        self.delete_btn = ctk.CTkButton(sidebar, text="Remove Podcast", fg_color="#c0392b",
                                         hover_color="#e74c3c", height=30,
                                         command=self._delete_podcast, state="disabled")
        self.delete_btn.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))

        return sidebar

    def _build_main_area(self):
        main = ctk.CTkFrame(self._paned, corner_radius=0)
        main.pack(fill="both", expand=True)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Top bar: episode list header + pagination + transcribe button
        top_bar = ctk.CTkFrame(main, fg_color="transparent", height=40)
        top_bar.grid(row=0, column=0, sticky="ew")

        self.episode_header = ctk.CTkLabel(top_bar, text="Select a podcast to browse episodes",
                                            font=ctk.CTkFont(size=15, weight="bold"))
        self.episode_header.pack(side="left")

        # Transcribe button
        self.transcribe_btn = ctk.CTkButton(top_bar, text="Transcribe", width=110,
                                              state="disabled", command=self._on_transcribe)
        self.transcribe_btn.pack(side="right", padx=(8, 0))

        # Pagination
        self.page_label = ctk.CTkLabel(top_bar, text="")
        self.page_label.pack(side="right", padx=6)

        self.next_btn = ctk.CTkButton(top_bar, text=">", width=32, height=28,
                                        command=self._next_page, state="disabled")
        self.next_btn.pack(side="right")
        self.prev_btn = ctk.CTkButton(top_bar, text="<", width=32, height=28,
                                        command=self._prev_page, state="disabled")
        self.prev_btn.pack(side="right", padx=(0, 4))

        # Episodes + progress
        pane = ctk.CTkFrame(main, fg_color="transparent")
        pane.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        pane.grid_rowconfigure(0, weight=1)
        pane.grid_rowconfigure(1, weight=0)  # progress bar row
        pane.grid_columnconfigure(0, weight=1)

        # Episode table (scrollable frame)
        self.episode_list_frame = ctk.CTkScrollableFrame(pane)
        self.episode_list_frame.grid(row=0, column=0, sticky="nsew")
        self.episode_list_frame.grid_columnconfigure(0, weight=1)

        self._episode_widgets = []
        self._selected_episode_idx = None

        # Progress area
        progress_frame = ctk.CTkFrame(pane, fg_color="transparent", height=52)
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(progress_frame, height=14)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(progress_frame, text="", font=ctk.CTkFont(size=12))
        self.progress_label.grid(row=0, column=1)

        self.progress_detail = ctk.CTkLabel(progress_frame, text="", font=ctk.CTkFont(size=11),
                                             text_color="gray")
        self.progress_detail.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        return main

    def _build_transcript_panel(self):
        """Build the transcript panel (3rd column, initially hidden)."""
        panel = ctk.CTkFrame(self._paned, width=180, corner_radius=0)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        self._transcript_title = ctk.CTkLabel(header, text="Transcript",
                                               font=ctk.CTkFont(size=15, weight="bold"))
        self._transcript_title.pack(side="left")

        # Transcript textbox (read-only)
        self.transcript_box = ctk.CTkTextbox(panel, font=ctk.CTkFont(family="Menlo", size=13),
                                               wrap="word", state="disabled")
        self.transcript_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 4))

        # Bottom bar: Copy + Close
        bottom = ctk.CTkFrame(panel, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 12))

        ctk.CTkButton(bottom, text="Copy", width=70, command=self._copy_transcript
                       ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(bottom, text="Close", width=70, fg_color="gray",
                       hover_color="#555", command=self._close_transcript_panel).pack(side="left")

        return panel

    def _show_transcript_panel(self):
        """Show the transcript panel as 3rd pane if not already visible."""
        if self._transcript_panel is None:
            self._transcript_panel = self._build_transcript_panel()
        if not self._transcript_panel_visible:
            self._paned.add(self._transcript_panel, minsize=120, width=180)
            self._transcript_panel_visible = True

    def _close_transcript_panel(self):
        """Remove the transcript panel from the PanedWindow."""
        if self._transcript_panel_visible and self._transcript_panel is not None:
            self._paned.forget(self._transcript_panel)
            self._transcript_panel_visible = False

    def _copy_transcript(self):
        """Copy transcript text to clipboard."""
        self.transcript_box.configure(state="normal")
        text = self.transcript_box.get("1.0", "end").strip()
        self.transcript_box.configure(state="disabled")
        self.clipboard_clear()
        self.clipboard_append(text)

    def _show_podcast_info(self, podcast_id):
        """Show podcast info in a modal dialog (right-click handler)."""
        podcast = db.get_podcast(podcast_id)
        if not podcast:
            return

        dialog = ctk.CTkToplevel(self)
        dialog.withdraw()  # hide until positioned
        dialog.title(f"{podcast.get('title', 'Podcast')} — Info")
        dialog.geometry("500x400")
        dialog.resizable(True, True)
        dialog.transient(self)

        # Center on parent
        x = self.winfo_rootx() + (self.winfo_width() - 500) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 400) // 2
        dialog.geometry(f"500x400+{x}+{y}")

        frame = ctk.CTkFrame(dialog, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=16, pady=16)
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Top row: artwork + podcast name
        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # Use cached sidebar icon scaled up, or default
        artwork_img = None
        cached = self._podcast_icon_cache.get(podcast_id)
        if cached:
            # Re-fetch the full image from the URL for larger size
            image_url = podcast.get("image_url", "")
            if image_url:
                try:
                    resp = requests.get(image_url, timeout=10)
                    resp.raise_for_status()
                    artwork_img = ctk.CTkImage(light_image=Image.open(io.BytesIO(resp.content)), size=(48, 48))
                except Exception:
                    pass
        if artwork_img is None and ICON_PATH.exists():
            artwork_img = ctk.CTkImage(light_image=Image.open(ICON_PATH), size=(48, 48))

        icon_label = ctk.CTkLabel(top, text="")
        if artwork_img:
            icon_label.configure(image=artwork_img)
            icon_label._artwork = artwork_img  # prevent GC
        icon_label.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(top, text=podcast.get("title", ""), font=ctk.CTkFont(size=16, weight="bold"),
                     wraplength=380, justify="left").pack(side="left", fill="x", expand=True)

        # Separator
        ctk.CTkFrame(frame, height=2, fg_color="gray30").grid(row=1, column=0, sticky="ew")

        # Description (strip HTML)
        desc_box = ctk.CTkTextbox(frame, font=ctk.CTkFont(size=13), wrap="word", state="disabled")
        desc_box.grid(row=2, column=0, sticky="nsew", pady=(8, 8))
        desc = _strip_html(podcast.get("description", ""))
        desc_box.configure(state="normal")
        if desc:
            desc_box.insert("1.0", desc)
        desc_box.configure(state="disabled")

        # Bottom: Close button
        ctk.CTkButton(frame, text="Close", width=80, command=dialog.destroy
                       ).grid(row=3, column=0, sticky="e")

        # Show centered, then grab focus
        dialog.deiconify()
        dialog.grab_set()

    # ─── Podcast list management ─────────────────────────────────────

    def _load_podcasts(self):
        for w in self.podcast_list_frame.winfo_children():
            w.destroy()

        podcasts = db.get_all_podcasts()
        if not podcasts:
            ctk.CTkLabel(self.podcast_list_frame, text="No podcasts yet.\nClick + to add one.",
                         text_color="gray", justify="center").pack(pady=40)
            return

        # Default icon for podcasts without artwork
        if not hasattr(self, "_default_podcast_icon") and ICON_PATH.exists():
            self._default_podcast_icon = _make_square_icon(Image.open(ICON_PATH))

        self._podcast_buttons = []
        for p in podcasts:
            # Use cached icon or default
            icon = self._podcast_icon_cache.get(
                p["id"], getattr(self, "_default_podcast_icon", None))

            row = ctk.CTkFrame(self.podcast_list_frame, fg_color="transparent",
                                height=36, cursor="hand2")
            row.pack(fill="x", pady=1)
            row.grid_columnconfigure(1, weight=1)

            icon_label = ctk.CTkLabel(row, text="", image=icon,
                                       width=SIDEBAR_ICON_SIZE, height=SIDEBAR_ICON_SIZE)
            icon_label.grid(row=0, column=0, padx=(4, 6), pady=4)

            name_label = ctk.CTkLabel(row, text=p["title"], anchor="w",
                                       font=ctk.CTkFont(size=14, weight="bold"))
            name_label.grid(row=0, column=1, sticky="w", pady=4)

            # Click handler
            cmd = lambda e=None, pid=p["id"], title=p["title"]: self._select_podcast(pid, title)
            for w in (row, icon_label, name_label):
                w.bind("<Button-1>", cmd)

            # Right-click to show podcast info
            for w in (row, icon_label, name_label):
                w.bind("<Button-2>", lambda e, pid=p["id"]: self._show_podcast_info(pid))
                w.bind("<Button-3>", lambda e, pid=p["id"]: self._show_podcast_info(pid))

            # Store references for highlight and icon update
            row._icon_label = icon_label
            self._podcast_buttons.append((p["id"], row))

            # Load artwork in background if not cached
            image_url = p.get("image_url", "")
            if image_url and p["id"] not in self._podcast_icon_cache:
                threading.Thread(
                    target=self._load_podcast_icon,
                    args=(p["id"], image_url),
                    daemon=True,
                ).start()

    def _load_podcast_icon(self, podcast_id, image_url):
        """Background: download podcast artwork and update the sidebar icon."""
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            icon = _make_square_icon(Image.open(io.BytesIO(resp.content)))
            self._podcast_icon_cache[podcast_id] = icon
            self.after(0, lambda: self._update_podcast_button_icon(podcast_id, icon))
        except Exception:
            pass

    def _update_podcast_button_icon(self, podcast_id, icon):
        """Update a podcast row's icon after background load."""
        for pid, row in self._podcast_buttons:
            if pid == podcast_id:
                try:
                    row._icon_label.configure(image=icon)
                except Exception:
                    pass
                break

    def _select_podcast(self, podcast_id, title):
        self.selected_podcast_id = podcast_id
        self.episode_page = 0
        self.delete_btn.configure(state="normal")

        # Highlight selected
        for pid, btn in self._podcast_buttons:
            if pid == podcast_id:
                btn.configure(fg_color=("gray75", "gray30"))
            else:
                btn.configure(fg_color="transparent")

        self.episode_header.configure(text=title)
        self._load_episodes()

    def _open_add_dialog(self):
        AddPodcastDialog(self, self._load_podcasts)

    def _delete_podcast(self):
        if self.selected_podcast_id is None:
            return
        db.delete_podcast(self.selected_podcast_id)
        self.selected_podcast_id = None
        self.delete_btn.configure(state="disabled")
        self.episode_header.configure(text="Select a podcast to browse episodes")
        self._clear_episodes()
        self._set_transcript_text("")
        self._load_podcasts()

    def _preload_models(self):
        """Pre-load base and small models in background threads."""
        default_engine = self.engine_var.get()
        for model_name in ("base", "small"):
            threading.Thread(
                target=self._preload_one_model,
                args=(default_engine, model_name),
                daemon=True,
            ).start()

    def _preload_one_model(self, engine, model_name):
        """Load a single model into the cache, updating progress detail."""
        def _progress(msg):
            self._update_progress(0, "", msg)
        try:
            _get_or_load_model(engine, model_name, progress_fn=_progress)
            self._update_progress(0, "", f"Model '{model_name}' ready")
        except Exception:
            pass

    def _refresh_all_podcasts(self):
        """Background: refresh episodes from all podcast RSS feeds."""
        self._update_progress(0, "", "Refreshing feeds...")
        podcasts = db.get_all_podcasts()
        for p in podcasts:
            try:
                data = fetch_feed(p["rss_url"])
                db.refresh_podcast_episodes(p["id"], p["rss_url"], data["episodes"])
                db.update_podcast_meta(
                    p["id"],
                    data.get("description", ""),
                    data.get("image_url", ""),
                )
            except Exception:
                pass
        self.after(0, self._load_podcasts)
        if self.selected_podcast_id is not None:
            self.after(0, self._load_episodes)
        self._update_progress(0, "", "Feeds updated")

    # ─── Episode list ────────────────────────────────────────────────

    def _episodes_per_page(self):
        """Calculate how many episode rows fit based on window height."""
        try:
            win_h = self.winfo_height()
        except Exception:
            return MIN_EPISODES_PER_PAGE
        if win_h <= 1:
            return MIN_EPISODES_PER_PAGE
        usable = win_h - EPISODE_OVERHEAD
        count = max(MIN_EPISODES_PER_PAGE, usable // EPISODE_ROW_HEIGHT)
        return count

    def _on_resize(self, event):
        """Debounced handler: reload episodes if page size changed after resize."""
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(200, self._check_page_size)

    def _check_page_size(self):
        """Reload episodes if the computed page size changed."""
        self._resize_after_id = None
        new_count = self._episodes_per_page()
        if new_count != self._last_eps_per_page and self.selected_podcast_id is not None:
            self._last_eps_per_page = new_count
            self.episode_page = 0
            self._load_episodes()

    def _load_episodes(self):
        self._clear_episodes()
        if self.selected_podcast_id is None:
            return

        page_size = self._episodes_per_page()
        self._last_eps_per_page = page_size

        self.episode_total = db.count_episodes(self.selected_podcast_id)
        episodes = db.get_episodes(self.selected_podcast_id,
                                    limit=page_size,
                                    offset=self.episode_page * page_size)

        total_pages = max(1, (self.episode_total + page_size - 1) // page_size)
        current_page = self.episode_page + 1
        self.page_label.configure(text=f"{current_page}/{total_pages}")
        self.prev_btn.configure(state="normal" if self.episode_page > 0 else "disabled")
        self.next_btn.configure(state="normal" if current_page < total_pages else "disabled")

        if not episodes:
            ctk.CTkLabel(self.episode_list_frame, text="No episodes found.",
                         text_color="gray").grid(row=0, column=0, pady=20)
            return

        # Column headers: #, Title, Published, Duration, Status
        hdr = ctk.CTkFrame(self.episode_list_frame, fg_color="transparent", height=24)
        hdr.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 2))
        hdr.grid_columnconfigure(0, weight=0, minsize=30)
        hdr.grid_columnconfigure(1, weight=1)
        hdr.grid_columnconfigure(2, weight=0, minsize=120)
        hdr.grid_columnconfigure(3, weight=0, minsize=100)
        hdr.grid_columnconfigure(4, weight=0, minsize=60)

        ctk.CTkLabel(hdr, text="#", font=ctk.CTkFont(size=11, weight="bold"),
                     width=30).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(hdr, text="Title", font=ctk.CTkFont(size=11, weight="bold"),
                     anchor="w").grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(hdr, text="Published", font=ctk.CTkFont(size=11, weight="bold"),
                     width=120).grid(row=0, column=2)
        ctk.CTkLabel(hdr, text="Duration", font=ctk.CTkFont(size=11, weight="bold"),
                     width=100).grid(row=0, column=3)
        ctk.CTkLabel(hdr, text="Status", font=ctk.CTkFont(size=11, weight="bold"),
                     width=60).grid(row=0, column=4)

        self._episode_data = episodes
        self._episode_widgets = []
        self._selected_episode_idx = None

        for i, ep in enumerate(episodes):
            has_transcript = db.get_transcript(ep["id"]) is not None
            row_num = i + 1
            global_num = self.episode_page * page_size + i + 1

            row = ctk.CTkFrame(self.episode_list_frame, fg_color="transparent",
                                cursor="hand2")
            row.grid(row=row_num, column=0, sticky="ew", padx=4, pady=(2, 2))
            row.grid_columnconfigure(0, weight=0, minsize=30)
            row.grid_columnconfigure(1, weight=1)
            row.grid_columnconfigure(2, weight=0, minsize=120)
            row.grid_columnconfigure(3, weight=0, minsize=100)
            row.grid_columnconfigure(4, weight=0, minsize=60)

            ctk.CTkLabel(row, text=str(global_num), width=30,
                         font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
            title_label = ctk.CTkLabel(row, text=ep["title"], anchor="w",
                                        font=ctk.CTkFont(size=12, weight="bold"))
            title_label.grid(row=0, column=1, sticky="w", padx=(4, 8))
            ctk.CTkLabel(row, text=_format_publish_date(ep.get("published", "")), width=120,
                         font=ctk.CTkFont(size=12), text_color="gray").grid(row=0, column=2)
            ctk.CTkLabel(row, text=_normalize_duration(ep.get("duration", "")), width=100,
                         font=ctk.CTkFont(size=12)).grid(row=0, column=3)

            status_text = "Done" if has_transcript else ""
            status_color = "#2ecc71" if has_transcript else "gray"
            ctk.CTkLabel(row, text=status_text, width=60, text_color=status_color,
                         font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=4)

            # Episode blurb/summary below title
            summary = _strip_html(ep.get("summary", "")).strip()
            if summary:
                if len(summary) > 150:
                    summary = summary[:150].rstrip() + "..."
                blurb_label = ctk.CTkLabel(row, text=summary, anchor="w",
                                            font=ctk.CTkFont(size=11), text_color="gray",
                                            wraplength=500, justify="left")
                blurb_label.grid(row=1, column=1, columnspan=4, sticky="w", padx=(4, 8))

            # Bind click on the whole row
            for widget in [row] + list(row.winfo_children()):
                widget.bind("<Button-1>", lambda e, idx=i: self._select_episode(idx))

            self._episode_widgets.append(row)

    def _clear_episodes(self):
        for w in self.episode_list_frame.winfo_children():
            w.destroy()
        self._episode_widgets = []
        self._selected_episode_idx = None
        self.transcribe_btn.configure(state="disabled")
        self.page_label.configure(text="")
        self.prev_btn.configure(state="disabled")
        self.next_btn.configure(state="disabled")

    def _select_episode(self, idx):
        if self._transcribing:
            return
        self._selected_episode_idx = idx

        # Highlight
        for i, w in enumerate(self._episode_widgets):
            if i == idx:
                w.configure(fg_color=("gray80", "gray25"))
            else:
                w.configure(fg_color="transparent")

        self.transcribe_btn.configure(state="normal")

        # Show existing transcript if available
        ep = self._episode_data[idx]
        transcript = db.get_transcript(ep["id"])
        if transcript:
            self._set_transcript_text(transcript["text"])
        else:
            self._set_transcript_text(f"No transcript yet.\n\nClick 'Transcribe' to generate one "
                                      f"using the '{self.model_var.get()}' model.")

    def _next_page(self):
        self.episode_page += 1
        self._load_episodes()

    def _prev_page(self):
        if self.episode_page > 0:
            self.episode_page -= 1
            self._load_episodes()

    # ─── Transcription ───────────────────────────────────────────────

    def _on_transcribe(self):
        if self._selected_episode_idx is None or self._transcribing:
            return

        ep = self._episode_data[self._selected_episode_idx]
        model_name = self.model_var.get()
        do_diarize = self.diarize_var.get()
        engine = self.engine_var.get()

        self._transcribing = True
        self._cancel_event.clear()
        self.transcribe_btn.configure(
            state="normal", text="Stop", fg_color="#c0392b",
            hover_color="#e74c3c", command=self._on_stop)
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting...")
        self.progress_detail.configure(text=f"{engine} '{model_name}'")
        self._set_transcript_text("")

        threading.Thread(target=self._transcribe_worker,
                         args=(ep, model_name, do_diarize, engine), daemon=True).start()

    def _on_stop(self):
        """Signal the worker thread to stop."""
        self._cancel_event.set()
        self.transcribe_btn.configure(state="disabled", text="Stopping...")

    def _check_cancelled(self):
        """Raise _TranscriptionCancelled if the user pressed Stop."""
        if self._cancel_event.is_set():
            raise _TranscriptionCancelled()

    def _transcribe_worker(self, episode, model_name, do_diarize, engine="whisper"):
        job_t0 = time.monotonic()

        try:
            audio_url = episode["audio_url"]
            extension = Path(audio_url.split("?")[0]).suffix or ".mp3"

            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # ── Stage 1: Download ────────────────────────────────
                stage_t0 = time.monotonic()
                self._update_progress(0.05, "Stage 1/4: Downloading audio")
                response = requests.get(audio_url, stream=True, timeout=30)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                total_mb = total / (1024 * 1024) if total else 0

                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        self._check_cancelled()
                        f.write(chunk)
                        downloaded += len(chunk)
                        dl_mb = downloaded / (1024 * 1024)
                        if total:
                            dl_pct = downloaded / total
                            elapsed = time.monotonic() - stage_t0
                            eta = _estimate_remaining(elapsed, dl_pct)
                            eta_str = f" — ~{eta} remaining" if eta else ""
                            self._update_progress(
                                0.05 + dl_pct * 0.25,
                                "Stage 1/4: Downloading audio",
                                f"{dl_mb:.1f} / {total_mb:.1f} MB ({dl_pct * 100:.0f}%){eta_str}")
                        else:
                            self._update_progress(
                                0.15,
                                "Stage 1/4: Downloading audio",
                                f"{dl_mb:.1f} MB")

                self._check_cancelled()

                # ── Stage 2: Load model (cached) ─────────────────────
                stage_t0 = time.monotonic()
                if engine == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
                    self._update_progress(0.32, "Stage 2/4: Loading model",
                                          f"faster-whisper '{model_name}' (int8, cpu)")
                    fw_model = _get_or_load_model(engine, model_name)
                    self._check_cancelled()

                    # ── Stage 3: Transcribe (faster-whisper) ─────────
                    stage_t0 = time.monotonic()
                    self._update_progress(0.35, "Stage 3/4: Transcribing",
                                          f"faster-whisper '{model_name}' — starting...")
                    fw_segments_iter, info = fw_model.transcribe(str(tmp_path))
                    audio_duration = info.duration
                    duration_str = format_timestamp(audio_duration)
                    segments = []
                    for seg in fw_segments_iter:
                        self._check_cancelled()
                        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
                        seg_pct = min(seg.end / audio_duration, 1.0) if audio_duration else 0
                        progress = 0.35 + seg_pct * 0.50
                        elapsed = time.monotonic() - stage_t0
                        eta = _estimate_remaining(elapsed, seg_pct)
                        eta_str = f" — ~{eta} remaining" if eta else ""
                        self._update_progress(
                            progress, "Stage 3/4: Transcribing",
                            f"faster-whisper '{model_name}' — "
                            f"{len(segments)} segments | "
                            f"{format_timestamp(seg.end)} / {duration_str}"
                            f"{eta_str}")
                    transcribe_elapsed = time.monotonic() - stage_t0
                    self._update_progress(0.85, "Transcription complete",
                                          f"faster-whisper '{model_name}' — "
                                          f"{len(segments)} segments | {duration_str} "
                                          f"in {_format_duration(transcribe_elapsed)}")
                else:
                    self._update_progress(0.32, "Stage 2/4: Loading model",
                                          f"whisper '{model_name}'")
                    model = _get_or_load_model(engine, model_name)
                    self._check_cancelled()

                    # ── Stage 3: Transcribe (whisper, per-segment via stdout capture)
                    stage_t0 = time.monotonic()
                    audio = whisper.audio.load_audio(str(tmp_path))
                    audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
                    duration_str = format_timestamp(audio_duration)
                    self._update_progress(0.35, "Stage 3/4: Transcribing",
                                          f"whisper '{model_name}' — starting...")
                    progress_writer = _WhisperProgressWriter(
                        audio_duration, stage_t0, self._update_progress,
                        model_name, self._cancel_event)
                    old_stdout = sys.stdout
                    try:
                        sys.stdout = progress_writer
                        result = model.transcribe(str(tmp_path), verbose=True, fp16=False)
                    finally:
                        sys.stdout = old_stdout
                    segments = result["segments"]
                    transcribe_elapsed = time.monotonic() - stage_t0
                    self._update_progress(0.85, "Transcription complete",
                                          f"whisper '{model_name}' — {len(segments)} segments "
                                          f"| {duration_str} "
                                          f"in {_format_duration(transcribe_elapsed)}")

                self._check_cancelled()

                # ── Stage 4: Diarize ─────────────────────────────────
                has_speakers = False
                if do_diarize and DIARIZATION_AVAILABLE:
                    hf_token = os.getenv("HF_TOKEN")
                    if hf_token and hf_token != "your_huggingface_token_here":
                        stage_t0 = time.monotonic()
                        self._update_progress(0.88, "Stage 4/4: Speaker diarization",
                                              "pyannote.audio — loading pipeline...")
                        pipeline = DiarizationPipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=hf_token,
                        )
                        self._check_cancelled()
                        self._update_progress(0.90, "Stage 4/4: Speaker diarization",
                                              f"pyannote.audio — analyzing speakers "
                                              f"(elapsed {_format_duration(time.monotonic() - stage_t0)})...")
                        diarization = pipeline(str(tmp_path))
                        self._check_cancelled()

                        # Assign speakers
                        speaker_timeline = []
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            speaker_timeline.append({
                                "start": turn.start, "end": turn.end, "speaker": speaker,
                            })

                        speaker_names = {}
                        name_counter = 1
                        for seg in segments:
                            best_speaker, best_overlap = None, 0
                            for sp in speaker_timeline:
                                overlap = max(0, min(seg["end"], sp["end"]) - max(seg["start"], sp["start"]))
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_speaker = sp["speaker"]
                            if best_speaker and best_speaker not in speaker_names:
                                speaker_names[best_speaker] = f"Speaker {name_counter}"
                                name_counter += 1
                            seg["speaker"] = speaker_names.get(best_speaker, "Unknown") if best_speaker else "Unknown"
                        has_speakers = True

                # ── Format + save ────────────────────────────────────
                self._update_progress(0.95, "Formatting transcript...", "")
                lines = []
                if has_speakers:
                    current_speaker = None
                    for seg in segments:
                        speaker = seg.get("speaker", "Unknown")
                        ts = format_timestamp(seg["start"])
                        text = seg["text"].strip()
                        if speaker != current_speaker:
                            current_speaker = speaker
                            lines.append(f"\n[{speaker}] ({ts})")
                        lines.append(text)
                else:
                    for seg in segments:
                        ts = format_timestamp(seg["start"])
                        lines.append(f"[{ts}] {seg['text'].strip()}")

                transcript_text = "\n".join(lines)

                # Save to DB
                db.save_transcript(episode["id"], transcript_text, model_name, diarized=has_speakers)

                total_elapsed = time.monotonic() - job_t0
                self._update_progress(1.0, "Done!",
                                      f"Total time: {_format_duration(total_elapsed)}")
                self.after(0, lambda: self._set_transcript_text(transcript_text))
                self.after(0, self._load_episodes)

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except _TranscriptionCancelled:
            elapsed = time.monotonic() - job_t0
            self._update_progress(0, "Cancelled",
                                  f"Stopped after {_format_duration(elapsed)}")
            self.after(0, lambda: self._set_transcript_text("Transcription cancelled."))
        except Exception as e:
            self.after(0, lambda: self._set_transcript_text(f"Error: {e}"))
            self._update_progress(0, f"Error: {e}", "")
        finally:
            self.after(0, self._transcription_done)

    def _update_progress(self, value, text, detail=""):
        self.after(0, lambda: self.progress_bar.set(value))
        self.after(0, lambda: self.progress_label.configure(text=text))
        self.after(0, lambda: self.progress_detail.configure(text=detail))

    def _transcription_done(self):
        self._transcribing = False
        self.transcribe_btn.configure(
            state="normal", text="Transcribe", fg_color=("#3a7ebf", "#1f538d"),
            hover_color=("#325882", "#14375e"), command=self._on_transcribe)

    # ─── Transcript display ──────────────────────────────────────────

    def _set_transcript_text(self, text):
        self._show_transcript_panel()
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        if text:
            self.transcript_box.insert("1.0", text)
        self.transcript_box.configure(state="disabled")


def main():
    app = PodcastApp()
    app.mainloop()


if __name__ == "__main__":
    main()
