"""Shared checkout paths; supplied fixtures stay at the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = ROOT / "data" / "helpdesk.json"
