"""SQLite database layer for Podcast Transcriber GUI."""

import email.utils
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = os.getenv("DB_PATH", str(Path(__file__).parent / "podcasts.db"))


def _parse_published_date(raw):
    """Convert RFC 2822 date string to ISO 8601 for sortable storage. Returns '' on failure."""
    if not raw:
        return ""
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        return dt.isoformat()
    except Exception:
        return ""


def get_connection(db_path=None):
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path=None):
    """Create tables if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS podcasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rss_url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            author TEXT DEFAULT '',
            description TEXT DEFAULT '',
            image_url TEXT DEFAULT '',
            added_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            podcast_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            published TEXT DEFAULT '',
            published_at TEXT DEFAULT '',
            summary TEXT DEFAULT '',
            audio_url TEXT NOT NULL,
            duration TEXT DEFAULT '',
            FOREIGN KEY (podcast_id) REFERENCES podcasts(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER UNIQUE NOT NULL,
            text TEXT NOT NULL,
            model_used TEXT NOT NULL,
            diarized INTEGER DEFAULT 0,
            transcribed_at TEXT NOT NULL,
            FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER UNIQUE NOT NULL,
            text TEXT NOT NULL,
            prompt_used TEXT NOT NULL,
            model_used TEXT NOT NULL,
            analyzed_at TEXT NOT NULL,
            FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
        );
    """)
    # Migrate existing databases: add new columns if missing
    try:
        conn.execute("ALTER TABLE podcasts ADD COLUMN description TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE podcasts ADD COLUMN image_url TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE episodes ADD COLUMN published_at TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    # Backfill published_at for existing rows that have published but no published_at
    rows = conn.execute(
        "SELECT id, published FROM episodes WHERE published != '' AND (published_at IS NULL OR published_at = '')"
    ).fetchall()
    for row in rows:
        iso = _parse_published_date(row["published"])
        if iso:
            conn.execute("UPDATE episodes SET published_at = ? WHERE id = ?", (iso, row["id"]))
    conn.commit()
    conn.close()


# --- Podcast CRUD ---

def add_podcast(rss_url, title, author, episodes, description="", image_url=""):
    """Insert a podcast and its episodes. Returns podcast id."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO podcasts (rss_url, title, author, description, image_url, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rss_url, title, author, description, image_url, datetime.now().isoformat()),
        )
        podcast_id = cursor.lastrowid

        for ep in episodes:
            published = ep.get("published", "")
            conn.execute(
                "INSERT INTO episodes (podcast_id, title, published, published_at, summary, audio_url, duration) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (podcast_id, ep["title"], published, _parse_published_date(published),
                 ep.get("summary", ""), ep["audio_url"], ep.get("duration", "")),
            )
        conn.commit()
        return podcast_id
    except sqlite3.IntegrityError:
        conn.rollback()
        return None
    finally:
        conn.close()


def get_all_podcasts():
    """Return all podcasts."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM podcasts ORDER BY added_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_podcast(podcast_id):
    """Delete a podcast and its episodes/transcripts."""
    conn = get_connection()
    conn.execute("DELETE FROM podcasts WHERE id = ?", (podcast_id,))
    conn.commit()
    conn.close()


def get_podcast(podcast_id):
    """Return a single podcast by id, or None."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM podcasts WHERE id = ?", (podcast_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_podcast_meta(podcast_id, description, image_url):
    """Update description and image_url for a podcast."""
    conn = get_connection()
    conn.execute(
        "UPDATE podcasts SET description = ?, image_url = ? WHERE id = ?",
        (description, image_url, podcast_id),
    )
    conn.commit()
    conn.close()


def refresh_podcast_episodes(podcast_id, rss_url, episodes):
    """Replace all episodes for a podcast with fresh feed data. Keeps existing transcripts."""
    conn = get_connection()
    # Get existing episode audio_urls that have transcripts
    existing = conn.execute(
        "SELECT e.audio_url FROM episodes e "
        "JOIN transcripts t ON t.episode_id = e.id "
        "WHERE e.podcast_id = ?", (podcast_id,)
    ).fetchall()
    transcribed_urls = {r["audio_url"] for r in existing}

    # Delete episodes without transcripts, keep ones that have transcripts
    conn.execute(
        "DELETE FROM episodes WHERE podcast_id = ? AND id NOT IN "
        "(SELECT episode_id FROM transcripts)", (podcast_id,)
    )

    # Insert new episodes that don't already exist (by audio_url)
    existing_urls = {r["audio_url"] for r in conn.execute(
        "SELECT audio_url FROM episodes WHERE podcast_id = ?", (podcast_id,)
    ).fetchall()}

    for ep in episodes:
        if ep["audio_url"] not in existing_urls:
            published = ep.get("published", "")
            conn.execute(
                "INSERT INTO episodes (podcast_id, title, published, published_at, summary, audio_url, duration) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (podcast_id, ep["title"], published, _parse_published_date(published),
                 ep.get("summary", ""), ep["audio_url"], ep.get("duration", "")),
            )
    conn.commit()
    conn.close()


# --- Episode queries ---

def get_episodes(podcast_id, limit=20, offset=0):
    """Return paginated episodes for a podcast, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM episodes WHERE podcast_id = ? "
        "ORDER BY CASE WHEN published_at != '' THEN published_at ELSE id END DESC "
        "LIMIT ? OFFSET ?",
        (podcast_id, limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_episodes(podcast_id):
    """Return total episode count for a podcast."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM episodes WHERE podcast_id = ?", (podcast_id,)
    ).fetchone()
    conn.close()
    return row["cnt"]


# --- Transcript CRUD ---

def save_transcript(episode_id, text, model_used, diarized=False):
    """Save or replace a transcript for an episode."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO transcripts (episode_id, text, model_used, diarized, transcribed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (episode_id, text, model_used, int(diarized), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_transcript(episode_id):
    """Return transcript for an episode, or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM transcripts WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# --- Analysis CRUD ---

def save_analysis(episode_id, text, prompt_used, model_used):
    """Save or replace an analysis for an episode."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO analyses (episode_id, text, prompt_used, model_used, analyzed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (episode_id, text, prompt_used, model_used, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_analysis(episode_id):
    """Return analysis for an episode, or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM analyses WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None
