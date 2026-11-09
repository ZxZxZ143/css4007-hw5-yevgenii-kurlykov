"""Shared checkout paths; supplied fixtures stay at the repository root."""
from pathlib import Path, PureWindowsPath
import os

ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = ROOT / "data" / "helpdesk.json"


def repository_path(path: str | Path, *, base: Path = ROOT) -> Path:
    """Resolve native application paths against the explicit checkout base."""
    return (base / Path(path)).resolve()


def portable_path(path: Path, *, base: Path = ROOT) -> str:
    """Describe the actual file: repository-relative when possible, absolute otherwise."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def evidence_path(value: str, *, base: Path = ROOT) -> Path:
    """Read historical relative Windows paths without reinterpreting external paths."""
    native = Path(value)
    if native.is_absolute():
        return native.resolve()
    windows = PureWindowsPath(value)
    if windows.drive or windows.root:
        if os.name != "nt":
            raise ValueError("A foreign absolute Windows evidence path needs an explicit local mapping")
        return native.resolve()
    actual = repository_path(native, base=base)
    # On POSIX, an existing literal backslash filename is not a Windows path.
    if actual.exists() or "\\" not in value:
        return actual
    return repository_path(Path(*windows.parts), base=base)
