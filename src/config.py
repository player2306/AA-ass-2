"""Project paths and shared constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

DEFAULT_IMAGE = ROOT_DIR / "Broadway_tower_edit.jpg"
DEFAULT_OUTPUT = OUTPUT_DIR / "result.jpg"

MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 800
