"""Small, offline runtime utilities for reproducible future jobs."""
import hashlib
import json
import logging
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    """Return a UTC timestamp and random identifier for a run."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]


def sha256_file(path: Path) -> str:
    """Hash a file incrementally to bound memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_bytes(path: Path) -> int:
    """Return free disk bytes on the filesystem containing an existing path."""
    return shutil.disk_usage(path).free


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a JSON log entry; callers must not pass credentials or personal data."""
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))


def write_manifest(path: Path, manifest: dict[str, Any], *, dry_run: bool = False) -> None:
    """Create a JSON manifest exclusively, or validate serialization without writing."""
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if not dry_run:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
