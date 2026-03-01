"""Pure formatting utilities (no GUI dependencies)."""

import email.utils
import re

HTML_TAG_RE = re.compile(r'<[^>]+>')
HTML_ENTITY_RE = re.compile(r'&\w+;|&#\d+;')


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_duration(seconds):
    """Format seconds as human-readable duration (e.g. '2m 15s', '45s')."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def estimate_remaining(elapsed, fraction_done):
    """Estimate time remaining given elapsed time and fraction completed (0-1).

    Returns formatted string, or None if not enough data yet.
    """
    if fraction_done <= 0 or elapsed < 2:
        return None
    total_est = elapsed / fraction_done
    remaining = total_est - elapsed
    if remaining < 1:
        return None
    return format_duration(remaining)


def strip_html(text):
    """Remove HTML tags and collapse whitespace."""
    text = HTML_TAG_RE.sub(' ', text)
    text = HTML_ENTITY_RE.sub(' ', text)
    return ' '.join(text.split())


def normalize_duration(raw):
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


def format_publish_date(raw):
    """Format an RFC 2822 date string as 'Feb 24, 2026'. Falls back to raw string."""
    if not raw:
        return ""
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        return dt.strftime("%b %d, %Y")
    except Exception:
        return raw
