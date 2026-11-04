"""Offline checks of foundation utilities."""
import hashlib
import json
import logging

import pytest

from tecd_propensity.config import load_config
from tecd_propensity.runtime import (
    free_bytes,
    log_event,
    new_run_id,
    sha256_file,
    write_manifest,
)


def test_runtime(tmp_path, caplog):
    source = tmp_path / "input"
    source.write_bytes(b"fixture")
    assert sha256_file(source) == hashlib.sha256(b"fixture").hexdigest()
    assert free_bytes(tmp_path) > 0
    assert new_run_id() != new_run_id()
    path = tmp_path / "manifest.json"
    write_manifest(path, {"seed": 42}, dry_run=True)
    assert not path.exists()
    write_manifest(path, {"seed": 42})
    assert json.loads(path.read_text()) == {"seed": 42}
    with pytest.raises(FileExistsError):
        write_manifest(path, {})
    with caplog.at_level(logging.INFO):
        log_event(logging.getLogger("test"), "started", seed=42)
    assert json.loads(caplog.records[-1].message) == {"event": "started", "seed": 42}


def test_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")
    assert load_config(path) == {"seed": 42}
    path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(path)

