"""Small OS-backed locks shared by threads and independent local processes."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import time


@contextmanager
def file_lock(path: Path, *, timeout=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        if os.name == "nt":
            import msvcrt
            if file.seek(0, 2) == 0:
                file.write(b"\0")
                file.flush()
            def acquire():
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            def release():
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def acquire():
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def release():
                fcntl.flock(file, fcntl.LOCK_UN)
        deadline = time.monotonic() + timeout
        while True:
            try:
                acquire()
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("Local memory is busy; retry this operation") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            release()
