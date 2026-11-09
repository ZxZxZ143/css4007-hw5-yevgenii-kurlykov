"""Trusted user/thread namespacing and scoped SQLite inspection/reset."""
import hashlib
import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from ..paths import repository_path
from ..part2.runner import serializable
from .locking import file_lock
from .memory_store import validate_identifier


def checkpoint_key(user_id: str, thread_id: str) -> str:
    identity = [validate_identifier(user_id), validate_identifier(thread_id)]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest()


def checkpoint_config(user_id: str, thread_id: str) -> dict:
    return {"configurable": {"thread_id": checkpoint_key(user_id, thread_id)}}


def database_path(storage_dir: Path) -> Path:
    return repository_path(storage_dir) / "conversations.sqlite"


def inspect_thread(storage_dir: Path, user_id: str, thread_id: str) -> dict:
    config = checkpoint_config(user_id, thread_id)
    path = database_path(storage_dir)
    if not path.is_file():
        return {"exists": False, "checkpoint_key": config["configurable"]["thread_id"]}
    with SqliteSaver.from_conn_string(str(path)) as saver:
        record = saver.get_tuple(config)
    if record is None:
        return {"exists": False, "checkpoint_key": config["configurable"]["thread_id"]}
    state = record.checkpoint["channel_values"]
    return {"exists": True, "checkpoint_key": config["configurable"]["thread_id"],
            "message_count": len(state.get("messages", [])), "helpdesk": serializable(state.get("helpdesk", {})),
            "control": serializable(state.get("control", {}))}


def reset_thread(storage_dir: Path, user_id: str, thread_id: str):
    key = checkpoint_key(user_id, thread_id)
    directory = repository_path(storage_dir)
    with file_lock(directory / "locks" / f"{key}.lock"):
        path = database_path(directory)
        if path.is_file():
            with SqliteSaver.from_conn_string(str(path)) as saver:
                saver.delete_thread(key)
