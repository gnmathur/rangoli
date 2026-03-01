#!/usr/bin/env python3
"""
Rangoli — Podcast Transcriber

A CustomTkinter-based GUI for subscribing to podcasts, browsing episodes,
transcribing with Whisper, and viewing transcripts.

Author:  Gaurav Mathur
License: MIT
"""

import io
import multiprocessing.resource_tracker
import os
import ssl
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

# Suppress harmless semaphore leak warning on Ctrl+C during model preloading
warnings.filterwarnings("ignore", message="resource_tracker:.*semaphore", category=UserWarning)

from constants import (APP_NAME, APP_VERSION, APP_AUTHOR, APP_YEAR,
                       SIDEBAR_ICON_SIZE, WHISPER_MODELS, WHISPER_ENGINES,
                       EPISODE_ROW_HEIGHT, EPISODE_OVERHEAD, MIN_EPISODES_PER_PAGE,
                       ICON_PATH, DEFAULT_OPENAI_PROMPT)

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
import requests  # noqa: E402
import whisper  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import database as db  # noqa: E402
from utils import (format_timestamp, format_duration, estimate_remaining,  # noqa: E402
                   strip_html, normalize_duration, format_publish_date)
from feed import fetch_feed  # noqa: E402
from transcription import (FASTER_WHISPER_AVAILABLE, DIARIZATION_AVAILABLE,  # noqa: E402
                           OPENAI_AVAILABLE, TranscriptionCancelled,
                           get_or_load_model, WhisperProgressWriter,
                           DiarizationPipeline, openai)
from markdown_render import insert_markdown  # noqa: E402
from icons import make_square_icon  # noqa: E402
from dialogs import AddPodcastDialog  # noqa: E402

# SSL fix for macOS
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

load_dotenv()

# Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


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
        self._analyzing_ids = set()  # episode IDs currently being analyzed
        self._cancel_event = threading.Event()

        self._podcast_icon_cache = {}  # podcast_id -> CTkImage (24x24)

        self._last_eps_per_page = MIN_EPISODES_PER_PAGE

        # OpenAI configuration
        self._openai_prompt = os.getenv("OPENAI_PROMPT", DEFAULT_OPENAI_PROMPT)
        self._openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

        self._build_menubar()
        self._build_ui()
        self._build_analysis_panel()
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

        # AI menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_command(label="Edit Prompt Template...", command=self._edit_prompt_template)

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
        about.geometry("340x320")
        about.resizable(False, False)
        about.transient(self)
        about.grab_set()

        about.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - 340) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 320) // 2
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

        # Icon attribution
        attr_label = ctk.CTkLabel(frame, text="Icon: Freepik - Flaticon",
                                   font=ctk.CTkFont(size=10), text_color="#4a9eff",
                                   cursor="hand2")
        attr_label.pack(pady=(8, 0))
        attr_label.bind("<Button-1>", lambda e: __import__("webbrowser").open(
            "https://www.flaticon.com/free-icons/pattern"))

        ctk.CTkButton(frame, text="OK", width=80, command=about.destroy).pack(pady=(12, 0))

    def _set_appearance(self, mode):
        """Switch between dark and light mode."""
        ctk.set_appearance_mode(mode)
        sash_color = "#333333" if mode == "dark" else "#cccccc"
        bg_color = "#333333" if mode == "dark" else "#ebebeb"
        self._paned.configure(bg=bg_color, sashpad=0)

    def _edit_prompt_template(self):
        """Open a dialog to edit the OpenAI prompt template."""
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"{APP_NAME} — Edit Prompt Template")
        dialog.geometry("550x350")
        dialog.resizable(True, True)
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - 550) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 350) // 2
        dialog.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(dialog, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=16, pady=16)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="Prompt Template",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w")

        textbox = ctk.CTkTextbox(frame, font=ctk.CTkFont(family="Menlo", size=13), wrap="word")
        textbox.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        textbox.insert("1.0", self._openai_prompt)

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew")

        def _save():
            self._openai_prompt = textbox.get("1.0", "end-1c").strip()
            dialog.destroy()

        def _reset():
            textbox.delete("1.0", "end")
            textbox.insert("1.0", DEFAULT_OPENAI_PROMPT)

        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="gray",
                       hover_color="#555", command=dialog.destroy).pack(side="right", padx=(8, 0))
        ctk.CTkButton(btn_frame, text="Save", width=80, command=_save).pack(side="right")
        ctk.CTkButton(btn_frame, text="Reset Default", width=110, fg_color="gray",
                       hover_color="#555", command=_reset).pack(side="right", padx=(0, 8))

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
        self._paned.add(self._sidebar, minsize=220, width=340)

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

        # Settings: Model, Engine, Diarize + Add Podcast — grid-aligned
        settings_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        settings_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))
        settings_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings_frame, text="Model:", font=ctk.CTkFont(size=14, weight="bold")
                     ).grid(row=0, column=0, sticky="w", pady=2)
        default_model = os.getenv("WHISPER_MODEL", "base")
        self.model_var = ctk.StringVar(value=default_model)
        ctk.CTkOptionMenu(settings_frame, variable=self.model_var, values=WHISPER_MODELS,
                           width=140, height=28).grid(row=0, column=1, sticky="w", padx=(6, 0), pady=2)

        ctk.CTkLabel(settings_frame, text="Engine:", font=ctk.CTkFont(size=14, weight="bold")
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
        diarize_cb.grid(row=2, column=0, sticky="w", pady=2)
        if not DIARIZATION_AVAILABLE:
            diarize_cb.configure(state="disabled")

        ctk.CTkButton(settings_frame, text="Add Podcast", width=100, height=28,
                       command=self._open_add_dialog).grid(row=2, column=1, sticky="e", pady=2)

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

        # Top bar: episode list header + pagination
        top_bar = ctk.CTkFrame(main, fg_color="transparent", height=40)
        top_bar.grid(row=0, column=0, sticky="ew")

        self.episode_header = ctk.CTkLabel(top_bar, text="Select a podcast to browse episodes",
                                            font=ctk.CTkFont(size=15, weight="bold"))
        self.episode_header.pack(side="left")

        # Pagination
        self.page_label = ctk.CTkLabel(top_bar, text="")
        self.page_label.pack(side="right", padx=6)

        self.last_btn = ctk.CTkButton(top_bar, text="Last \u25b6\u25b6", width=56, height=28,
                                        cursor="arrow", command=self._last_page, state="disabled")
        self.last_btn.pack(side="right")
        self.next_btn = ctk.CTkButton(top_bar, text="Next \u25b6", width=56, height=28,
                                        cursor="arrow", command=self._next_page, state="disabled")
        self.next_btn.pack(side="right", padx=(0, 4))
        self.prev_btn = ctk.CTkButton(top_bar, text="\u25c0 Prev", width=56, height=28,
                                        cursor="arrow", command=self._prev_page, state="disabled")
        self.prev_btn.pack(side="right", padx=(0, 4))
        self.first_btn = ctk.CTkButton(top_bar, text="\u25c0\u25c0 First", width=56, height=28,
                                        cursor="arrow", command=self._first_page, state="disabled")
        self.first_btn.pack(side="right", padx=(0, 4))

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

        # Progress area with stop button
        progress_frame = ctk.CTkFrame(pane, fg_color="transparent", height=52)
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(progress_frame, height=14)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.progress_bar.set(0)

        self.stop_btn = ctk.CTkButton(progress_frame, text="Stop", width=76, height=30,
                                        fg_color="#c0392b", hover_color="#e74c3c",
                                        font=ctk.CTkFont(size=11),
                                        command=self._on_stop)
        self.stop_btn.grid(row=0, column=1)
        self.stop_btn.grid_remove()  # hidden by default

        self.progress_label = ctk.CTkLabel(progress_frame, text="", font=ctk.CTkFont(size=12))
        self.progress_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.progress_detail = ctk.CTkLabel(progress_frame, text="", font=ctk.CTkFont(size=11),
                                             text_color="gray")
        self.progress_detail.grid(row=2, column=0, sticky="w", pady=(2, 0))

        return main

    def _build_analysis_panel(self):
        """Build the 3rd column analysis panel (not added to PanedWindow yet)."""
        self._analysis_panel = ctk.CTkFrame(self._paned, corner_radius=0)
        self._analysis_panel.grid_rowconfigure(1, weight=1)
        self._analysis_panel.grid_columnconfigure(0, weight=1)

        # Row 0: header + close button
        header = ctk.CTkFrame(self._analysis_panel, fg_color="transparent", height=36)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="Analysis",
                     font=ctk.CTkFont(size=15, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(header, text="Close", width=60, height=28,
                       command=self._close_analysis_panel).grid(row=0, column=1, sticky="e")

        # Row 1: scrollable text
        self._analysis_textbox = ctk.CTkTextbox(self._analysis_panel,
                                                 font=ctk.CTkFont(family="Menlo", size=13),
                                                 wrap="word", state="disabled")
        self._analysis_textbox.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        # Row 2: copy button
        ctk.CTkButton(self._analysis_panel, text="Copy", width=80, height=28,
                       command=self._copy_analysis).grid(row=2, column=0, sticky="e", padx=8, pady=(0, 8))

        self._analysis_panel_visible = False

    def _show_analysis_panel(self, text):
        """Show the analysis panel with markdown-formatted text."""
        if not self._analysis_panel_visible:
            self._paned.add(self._analysis_panel, minsize=200, width=300)
            self._analysis_panel_visible = True

        self._analysis_textbox.configure(state="normal")
        self._analysis_textbox.delete("1.0", "end")
        insert_markdown(self._analysis_textbox, text)
        self._analysis_textbox.configure(state="disabled")

    def _close_analysis_panel(self):
        """Remove the analysis panel from the PanedWindow."""
        if self._analysis_panel_visible:
            self._paned.forget(self._analysis_panel)
            self._analysis_panel_visible = False

    def _copy_analysis(self):
        """Copy the analysis text to clipboard."""
        text = self._analysis_textbox.get("1.0", "end-1c")
        if text.strip():
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
        desc = strip_html(podcast.get("description", ""))
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
            self._default_podcast_icon = make_square_icon(Image.open(ICON_PATH))

        self._podcast_buttons = []
        for p in podcasts:
            # Use cached icon or default
            icon = self._podcast_icon_cache.get(
                p["id"], getattr(self, "_default_podcast_icon", None))

            row = ctk.CTkFrame(self.podcast_list_frame, fg_color="transparent",
                                height=36, cursor="arrow")
            row.pack(fill="x", pady=1)
            row.grid_columnconfigure(1, weight=1)

            icon_label = ctk.CTkLabel(row, text="", image=icon,
                                       width=SIDEBAR_ICON_SIZE, height=SIDEBAR_ICON_SIZE,
                                       cursor="arrow")
            icon_label.grid(row=0, column=0, padx=(4, 6), pady=4)

            name_label = ctk.CTkLabel(row, text=p["title"], anchor="w",
                                       font=ctk.CTkFont(size=14, weight="bold"),
                                       cursor="arrow")
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
            icon = make_square_icon(Image.open(io.BytesIO(resp.content)))
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
            get_or_load_model(engine, model_name, progress_fn=_progress)
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
        on_first = self.episode_page == 0
        on_last = current_page >= total_pages
        self.first_btn.configure(state="disabled" if on_first else "normal")
        self.prev_btn.configure(state="disabled" if on_first else "normal")
        self.next_btn.configure(state="disabled" if on_last else "normal")
        self.last_btn.configure(state="disabled" if on_last else "normal")

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
        hdr.grid_columnconfigure(4, weight=0, minsize=80)

        ctk.CTkLabel(hdr, text="#", font=ctk.CTkFont(size=11, weight="bold"),
                     width=30).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(hdr, text="Title", font=ctk.CTkFont(size=11, weight="bold"),
                     anchor="w").grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(hdr, text="Published", font=ctk.CTkFont(size=11, weight="bold"),
                     width=120).grid(row=0, column=2)
        ctk.CTkLabel(hdr, text="Duration", font=ctk.CTkFont(size=11, weight="bold"),
                     width=100).grid(row=0, column=3)
        ctk.CTkLabel(hdr, text="Status", font=ctk.CTkFont(size=11, weight="bold"),
                     width=80).grid(row=0, column=4)

        self._episode_data = episodes
        self._episode_widgets = []
        self._selected_episode_idx = None

        for i, ep in enumerate(episodes):
            has_transcript = db.get_transcript(ep["id"]) is not None
            has_analysis = db.get_analysis(ep["id"]) is not None
            row_num = i + 1
            global_num = self.episode_page * page_size + i + 1

            row = ctk.CTkFrame(self.episode_list_frame, fg_color="transparent",
                                cursor="arrow")
            row.grid(row=row_num, column=0, sticky="ew", padx=4, pady=(2, 2))
            row.grid_columnconfigure(0, weight=0, minsize=30)
            row.grid_columnconfigure(1, weight=1)
            row.grid_columnconfigure(2, weight=0, minsize=120)
            row.grid_columnconfigure(3, weight=0, minsize=100)
            row.grid_columnconfigure(4, weight=0, minsize=80)

            ctk.CTkLabel(row, text=str(global_num), width=30,
                         font=ctk.CTkFont(size=12), cursor="arrow").grid(row=0, column=0, sticky="w")
            title_label = ctk.CTkLabel(row, text=ep["title"], anchor="w",
                                        font=ctk.CTkFont(size=12, weight="bold"),
                                        cursor="arrow")
            title_label.grid(row=0, column=1, sticky="w", padx=(4, 8))
            ctk.CTkLabel(row, text=format_publish_date(ep.get("published", "")), width=120,
                         font=ctk.CTkFont(size=12), text_color="gray",
                         cursor="arrow").grid(row=0, column=2)
            ctk.CTkLabel(row, text=normalize_duration(ep.get("duration", "")), width=100,
                         font=ctk.CTkFont(size=12), cursor="arrow").grid(row=0, column=3)

            if ep["id"] in self._analyzing_ids:
                status_text, status_color = "Analyzing...", "#e67e22"
            elif has_analysis:
                status_text, status_color = "Analyzed", "#9b59b6"
            elif has_transcript:
                status_text, status_color = "Transcribed", "#2ecc71"
            else:
                status_text, status_color = "", "gray"
            ctk.CTkLabel(row, text=status_text, width=80, text_color=status_color,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         cursor="arrow").grid(row=0, column=4)

            # Episode blurb/summary below title
            summary = strip_html(ep.get("summary", "")).strip()
            if summary:
                if len(summary) > 150:
                    summary = summary[:150].rstrip() + "..."
                blurb_label = ctk.CTkLabel(row, text=summary, anchor="w",
                                            font=ctk.CTkFont(size=11), text_color="gray",
                                            wraplength=500, justify="left",
                                            cursor="arrow")
                blurb_label.grid(row=1, column=1, columnspan=4, sticky="w", padx=(4, 8))

            # Bind left-click to select, right-click for context menu
            for widget in [row] + list(row.winfo_children()):
                widget.bind("<Button-1>", lambda e, idx=i: self._select_episode(idx))
                widget.bind("<Button-2>", lambda e, idx=i: self._show_episode_menu(e, idx))
                widget.bind("<Button-3>", lambda e, idx=i: self._show_episode_menu(e, idx))

            self._episode_widgets.append(row)

    def _clear_episodes(self):
        for w in self.episode_list_frame.winfo_children():
            w.destroy()
        self._episode_widgets = []
        self._selected_episode_idx = None
        self.page_label.configure(text="")
        self.first_btn.configure(state="disabled")
        self.prev_btn.configure(state="disabled")
        self.next_btn.configure(state="disabled")
        self.last_btn.configure(state="disabled")

    def _select_episode(self, idx):
        self._selected_episode_idx = idx

        # Highlight
        for i, w in enumerate(self._episode_widgets):
            if i == idx:
                w.configure(fg_color=("gray80", "gray25"))
            else:
                w.configure(fg_color="transparent")

    def _show_episode_menu(self, event, idx):
        """Show a right-click context menu for an episode."""
        self._select_episode(idx)
        ep = self._episode_data[idx]
        has_transcript = db.get_transcript(ep["id"]) is not None
        has_analysis = db.get_analysis(ep["id"]) is not None
        has_api_key = bool(os.getenv("OPENAI_API_KEY"))

        menu = tk.Menu(self, tearoff=0)
        # Transcribe — disabled if already transcribed or currently transcribing
        if has_transcript or self._transcribing:
            menu.add_command(label="Transcribe", state="disabled")
        else:
            menu.add_command(label="Transcribe", command=self._on_transcribe)
        menu.add_separator()
        # Analyze with AI — disabled if no transcript, already analyzing this episode, or no API key
        ep_analyzing = ep["id"] in self._analyzing_ids
        if has_transcript and not ep_analyzing and has_api_key and OPENAI_AVAILABLE:
            menu.add_command(label="Analyze with AI",
                             command=lambda: self._analyze_episode(ep["id"]))
        else:
            menu.add_command(label="Analyze with AI", state="disabled")
        menu.add_separator()
        # Copy Transcript — disabled if no transcript
        if has_transcript:
            menu.add_command(label="Copy Transcript",
                             command=lambda: self._copy_episode_transcript(ep["id"]))
        else:
            menu.add_command(label="Copy Transcript", state="disabled")
        # Show Analysis — disabled if no analysis exists
        if has_analysis:
            menu.add_command(label="Show Analysis",
                             command=lambda: self._show_saved_analysis(ep["id"]))
        else:
            menu.add_command(label="Show Analysis", state="disabled")

        menu.tk_popup(event.x_root, event.y_root)

    def _copy_episode_transcript(self, episode_id):
        """Copy an episode's transcript to clipboard."""
        transcript = db.get_transcript(episode_id)
        if transcript:
            self.clipboard_clear()
            self.clipboard_append(transcript["text"])

    def _first_page(self):
        self.episode_page = 0
        self._load_episodes()

    def _prev_page(self):
        if self.episode_page > 0:
            self.episode_page -= 1
            self._load_episodes()

    def _next_page(self):
        self.episode_page += 1
        self._load_episodes()

    def _last_page(self):
        page_size = self._episodes_per_page()
        total_pages = max(1, (self.episode_total + page_size - 1) // page_size)
        self.episode_page = total_pages - 1
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
        self.stop_btn.grid()  # show stop button
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting...")
        self.progress_detail.configure(text=f"{engine} '{model_name}'")

        threading.Thread(target=self._transcribe_worker,
                         args=(ep, model_name, do_diarize, engine), daemon=True).start()

    def _on_stop(self):
        """Signal the worker thread to stop."""
        self._cancel_event.set()
        self.stop_btn.configure(state="disabled", text="Stopping...")

    def _check_cancelled(self):
        """Raise TranscriptionCancelled if the user pressed Stop."""
        if self._cancel_event.is_set():
            raise TranscriptionCancelled()

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
                            eta = estimate_remaining(elapsed, dl_pct)
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
                    fw_model = get_or_load_model(engine, model_name)
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
                        eta = estimate_remaining(elapsed, seg_pct)
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
                                          f"in {format_duration(transcribe_elapsed)}")
                else:
                    self._update_progress(0.32, "Stage 2/4: Loading model",
                                          f"whisper '{model_name}'")
                    model = get_or_load_model(engine, model_name)
                    self._check_cancelled()

                    # ── Stage 3: Transcribe (whisper, per-segment via stdout capture)
                    stage_t0 = time.monotonic()
                    audio = whisper.audio.load_audio(str(tmp_path))
                    audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
                    duration_str = format_timestamp(audio_duration)
                    self._update_progress(0.35, "Stage 3/4: Transcribing",
                                          f"whisper '{model_name}' — starting...")
                    progress_writer = WhisperProgressWriter(
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
                                          f"in {format_duration(transcribe_elapsed)}")

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
                                              f"(elapsed {format_duration(time.monotonic() - stage_t0)})...")
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
                                      f"Total time: {format_duration(total_elapsed)}")
                self.after(0, self._load_episodes)

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except TranscriptionCancelled:
            elapsed = time.monotonic() - job_t0
            self._update_progress(0, "Cancelled",
                                  f"Stopped after {format_duration(elapsed)}")
        except Exception as e:
            self._update_progress(0, f"Error: {e}", "")
        finally:
            self.after(0, self._transcription_done)

    # ─── OpenAI Analysis ────────────────────────────────────────────

    def _analyze_episode(self, episode_id):
        """Start analyzing an episode transcript with OpenAI in the background."""
        transcript = db.get_transcript(episode_id)
        if not transcript:
            return

        self._analyzing_ids.add(episode_id)
        self._load_episodes()

        threading.Thread(
            target=self._analyze_worker,
            args=(episode_id, transcript["text"]),
            daemon=True,
        ).start()

    def _analyze_worker(self, episode_id, transcript_text):
        """Background: call OpenAI API and save result."""
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": self._openai_prompt},
                    {"role": "user", "content": transcript_text},
                ],
            )
            analysis_text = response.choices[0].message.content

            db.save_analysis(episode_id, analysis_text, self._openai_prompt, self._openai_model)

            self.after(0, lambda: self._on_analysis_complete(analysis_text))
        except Exception as e:
            err_msg = str(e)
            if hasattr(e, 'message'):
                err_msg = e.message
            self.after(0, lambda: self._on_analysis_error(err_msg))
        finally:
            self.after(0, lambda: self._analysis_done(episode_id))

    def _on_analysis_complete(self, analysis_text):
        """Show a success dialog with the analysis result."""
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"{APP_NAME} — Analysis Complete")
        dialog.geometry("550x400")
        dialog.resizable(True, True)
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - 550) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 400) // 2
        dialog.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(dialog, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="Analysis Complete",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#2ecc71").grid(row=0, column=0, sticky="w", pady=(0, 8))

        msg_box = ctk.CTkTextbox(frame, font=ctk.CTkFont(family="Menlo", size=13),
                                  wrap="word", state="disabled")
        msg_box.grid(row=1, column=0, sticky="nsew")
        msg_box.configure(state="normal")
        insert_markdown(msg_box, analysis_text)
        msg_box.configure(state="disabled")

        ctk.CTkButton(frame, text="OK", width=80,
                       command=dialog.destroy).grid(row=2, column=0, sticky="e", pady=(12, 0))

    def _on_analysis_error(self, err_msg):
        """Show analysis error in a dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"{APP_NAME} — Analysis Error")
        dialog.geometry("480x200")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - 480) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 200) // 2
        dialog.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(dialog, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="Analysis Failed",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#e74c3c").grid(row=0, column=0, sticky="w", pady=(0, 8))

        msg_box = ctk.CTkTextbox(frame, font=ctk.CTkFont(size=13), wrap="word",
                                  height=80, state="disabled")
        msg_box.grid(row=1, column=0, sticky="nsew")
        msg_box.configure(state="normal")
        msg_box.insert("1.0", err_msg)
        msg_box.configure(state="disabled")

        ctk.CTkButton(frame, text="OK", width=80,
                       command=dialog.destroy).grid(row=2, column=0, sticky="e", pady=(12, 0))

    def _analysis_done(self, episode_id):
        self._analyzing_ids.discard(episode_id)
        self._load_episodes()

    def _show_saved_analysis(self, episode_id):
        """Show a previously saved analysis in the 3rd column panel."""
        analysis = db.get_analysis(episode_id)
        if analysis:
            self._show_analysis_panel(analysis["text"])

    def _update_progress(self, value, text, detail=""):
        self.after(0, lambda: self.progress_bar.set(value))
        self.after(0, lambda: self.progress_label.configure(text=text))
        self.after(0, lambda: self.progress_detail.configure(text=detail))

    def _transcription_done(self):
        self._transcribing = False
        self.stop_btn.configure(state="normal", text="Stop")
        self.stop_btn.grid_remove()  # hide stop button



def main():
    app = PodcastApp()
    app.mainloop()


if __name__ == "__main__":
    main()
