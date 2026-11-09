"""Strict per-user JSON preferences; conversation facts never belong in this file."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from ..paths import repository_path
from .locking import file_lock

ALLOWED = {"preferred_language": {"en", "ru", "kk"}, "response_style": {"concise", "detailed"}}
LANGUAGES = {"en": "en", "english": "en", "ru": "ru", "russian": "ru", "kk": "kk", "kz": "kk", "kazakh": "kk"}


class MemoryValidationError(ValueError):
    pass


class MemoryCorruptionError(ValueError):
    pass


def validate_identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise MemoryValidationError("Use a 1-128 character identifier containing letters, digits, dot, underscore or hyphen")
    return value


def normalize_preference(key, value) -> str:
    if not isinstance(key, str) or key not in ALLOWED or not isinstance(value, str) or len(value) > 32:
        raise MemoryValidationError("Unsupported preference key or value")
    normalized = value.strip().casefold()
    if key == "preferred_language":
        normalized = LANGUAGES.get(normalized, normalized)
    if normalized not in ALLOWED[key]:
        raise MemoryValidationError("Unsupported preference value")
    return normalized


class UserMemoryStore:
    def __init__(self, storage_dir: Path):
        self.directory = repository_path(storage_dir) / "users"

    def path_for(self, user_id: str) -> Path:
        user = validate_identifier(user_id)
        return self.directory / (hashlib.sha256(user.encode("utf-8")).hexdigest() + ".json")

    def _read(self, path: Path) -> dict:
        if not path.exists():
            return {"schema_version": 1, "preferences": {}}
        try:
            if path.stat().st_size > 4096:
                raise ValueError("Oversized memory")
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or set(document) != {"schema_version", "preferences"} or type(document["schema_version"]) is not int or document["schema_version"] != 1:
                raise ValueError("Invalid schema")
            preferences = document["preferences"]
            if not isinstance(preferences, dict) or any(normalize_preference(k, v) != v for k, v in preferences.items()):
                raise ValueError("Invalid preference document")
            return document
        except (ValueError, TypeError, UnicodeError):
            raise MemoryCorruptionError("User preference file is invalid; inspect or explicitly reset this user's memory") from None

    def load(self, user_id: str) -> dict:
        path = self.path_for(user_id)
        with file_lock(path.with_suffix(".lock")):
            return dict(self._read(path)["preferences"])

    def save(self, user_id: str, key: str, value: str) -> dict:
        normalized = normalize_preference(key, value)
        path = self.path_for(user_id)
        with file_lock(path.with_suffix(".lock")):
            document = self._read(path)
            if document["preferences"].get(key) == normalized:
                return {"success": True, "key": key, "value": normalized, "changed": False, "source": "user_json_memory"}
            document["preferences"][key] = normalized
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".preferences-", suffix=".tmp", delete=False) as file:
                    temporary = Path(file.name)
                    json.dump(document, file, indent=2, sort_keys=True)
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return {"success": True, "key": key, "value": normalized, "changed": True, "source": "user_json_memory"}

    def reset(self, user_id: str):
        path = self.path_for(user_id)
        with file_lock(path.with_suffix(".lock")):
            path.unlink(missing_ok=True)
