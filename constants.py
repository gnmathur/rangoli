"""Application-wide constants."""

from pathlib import Path

APP_NAME = "Rangoli"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Gaurav Mathur"
APP_YEAR = "2026"

SIDEBAR_ICON_SIZE = 24
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
WHISPER_ENGINES = ["whisper", "faster-whisper"]
EPISODE_ROW_HEIGHT = 52  # title + blurb + padding
EPISODE_OVERHEAD = 110  # top bar + progress area + padding
MIN_EPISODES_PER_PAGE = 5
ICON_PATH = Path(__file__).parent / "icon.png"

DEFAULT_OPENAI_PROMPT = (
    "Summarize this podcast transcript. Provide key topics discussed, "
    "main takeaways, and a brief overview."
)
