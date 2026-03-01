"""Application dialog windows."""

import threading

import customtkinter as ctk

import database as db
from constants import APP_NAME
from feed import fetch_feed


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
