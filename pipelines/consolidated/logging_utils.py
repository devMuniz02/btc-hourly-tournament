#!/usr/bin/env python3
"""
Logging helpers that tee stdout/stderr into the consolidated workflow log file.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Iterator, TextIO


class TeeStream:
    def __init__(self, original: TextIO, mirror: TextIO) -> None:
        self.original = original
        self.mirror = mirror

    def write(self, text: str) -> int:
        self.original.write(text)
        self.mirror.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.mirror.flush()

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def tee_output(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, handle)
        sys.stderr = TeeStream(original_stderr, handle)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
