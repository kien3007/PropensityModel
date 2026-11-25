"""
Deployment configuration guards: settings that must agree across files.
"""
import re
from pathlib import Path

from src.config.settings import settings

ROOT = Path(__file__).resolve().parents[1]


def test_serving_runs_one_worker_per_container():
    """Prometheus counters and the rate limiter are per process; several workers would split them."""
    cmd = next(line for line in (ROOT / "docker" / "Dockerfile.serving").read_text().splitlines()
               if line.startswith("CMD"))
    workers = re.search(r'"--workers",\s*"(\d+)"', cmd)
    assert workers is None or workers.group(1) == "1", cmd


def test_auc_drop_alert_matches_retraining_threshold():
    rules = (ROOT / "docker" / "prometheus" / "alert_rules.yml").read_text(encoding="utf-8")
    threshold = re.search(r"expr: propensity_realized_auc_drop > ([0-9.]+)", rules)
    assert threshold and float(threshold.group(1)) == settings.REALIZED_AUC_DROP_ALERT
