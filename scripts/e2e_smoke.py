"""
End-to-end smoke test of the docker-compose stack (Phase 6).

Brings the stack up in an isolated compose project with throw-away secrets and
non-default host ports, then checks the whole Lambda loop:

  build images -> services healthy -> batch scoring (ingest Bronze, score, publish to Redis)
  -> training (no promotion) -> ground-truth feedback -> API /ready + prediction from the
  Redis batch view -> speed event through Kafka -> consumer -> fused score changes
  -> /metrics exposes monitoring gauges -> Prometheus scrapes both targets, alert rules
  loaded -> Alertmanager ready -> tear down.

Usage:
    python -m scripts.e2e_smoke [--data real|synthetic] [--keep] [--skip-build]
  --data synthetic generates a Dunnhumby-shaped dataset (used in CI, no real data needed).
Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "propensity-e2e"
PORTS = {"SERVING_HOST_PORT": 18000, "REDIS_HOST_PORT": 16379, "KAFKA_HOST_PORT": 19092,
         "PROMETHEUS_HOST_PORT": 19090, "ALERTMANAGER_HOST_PORT": 19093, "GRAFANA_HOST_PORT": 13000}
SERVICES = ["redis", "kafka", "serving-api", "kafka-consumer", "scheduler", "alertmanager", "prometheus", "grafana"]


class Check:
    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}", flush=True)
        return ok

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.results)


def http(method: str, url: str, body: Optional[dict] = None, headers: Optional[dict] = None, timeout: float = 10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def wait_for(fn: Callable[[], bool], timeout: float, interval: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if fn():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def prepare_data(mode: str, workdir: Path) -> Path:
    if mode == "real":
        from src.config.settings import settings

        raw = Path(os.path.realpath(settings.DATA_RAW_DIR))
        if not (raw / "transaction_data.csv").exists():
            raise SystemExit(f"Real data not found at {raw}; use --data synthetic")
        return raw
    sys.path.insert(0, str(ROOT))
    from tests.fixtures.synthetic_dunnhumby import make_raw_tables, write_raw_csvs

    print("Generating synthetic Dunnhumby-shaped data (400 households x 711 days)...", flush=True)
    return write_raw_csvs(make_raw_tables(n_households=400, n_days=711, seed=11), workdir / "raw")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from src.config.settings import settings

    default_mode = "real" if (Path(settings.DATA_RAW_DIR) / "transaction_data.csv").exists() else "synthetic"
    parser.add_argument("--data", choices=["real", "synthetic"], default=default_mode)
    parser.add_argument("--keep", action="store_true", help="Leave the stack running for inspection")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="propensity_e2e_"))
    raw_dir = prepare_data(args.data, workdir)
    api_key = secrets.token_urlsafe(24)
    env_file = workdir / "e2e.env"
    env_file.write_text("\n".join([
        f"RAW_DATA_DIR={raw_dir}",
        f"SERVING_API_KEYS={api_key}",
        f"GRAFANA_ADMIN_PASSWORD={secrets.token_urlsafe(16)}",
        "STRICT_INFRA=true",
        *[f"{k}={v}" for k, v in PORTS.items()],
    ]) + "\n", encoding="utf-8")

    compose = ["docker", "compose", "-p", PROJECT, "--env-file", str(env_file), "-f", str(ROOT / "docker-compose.yml")]

    def run(cmd: List[str], timeout: int = 1800, check: bool = True) -> subprocess.CompletedProcess:
        print(f"$ {' '.join(cmd[len(compose):]) if cmd[:len(compose)] == compose else ' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
        return proc

    def pipeline(*cmd: str, timeout: int = 1800) -> subprocess.CompletedProcess:
        return run(compose + ["--profile", "tools", "run", "--rm", "batch-pipeline", *cmd], timeout=timeout)

    serving = f"http://127.0.0.1:{PORTS['SERVING_HOST_PORT']}"
    prometheus = f"http://127.0.0.1:{PORTS['PROMETHEUS_HOST_PORT']}"
    alertmanager = f"http://127.0.0.1:{PORTS['ALERTMANAGER_HOST_PORT']}"
    auth = {"X-API-Key": api_key}
    check = Check()
    started = time.time()
    print(f"E2E smoke test (data={args.data}, workdir={workdir})", flush=True)

    try:
        if not args.skip_build:
            check.record("images build", run(compose + ["--profile", "tools", "build"], timeout=3600).returncode == 0)
            if not check.ok:
                return 1
        up = run(compose + ["up", "-d", "--wait", "--wait-timeout", "300", *SERVICES], timeout=900)
        if not check.record("services healthy", up.returncode == 0, ", ".join(SERVICES)):
            return 1

        status, body = http("GET", f"{serving}/ready")
        check.record("serving not ready before batch", status == 503, f"HTTP {status}")

        batch = pipeline("python", "-m", "src.pipelines.batch_pipeline", "--cutoff", "711")
        check.record("batch scoring (Bronze ingest + score + Redis publish)", batch.returncode == 0)
        train = pipeline("python", "-m", "src.pipelines.train_pipeline", "--no-promote")
        check.record("training pipeline (no promotion)", train.returncode == 0)
        feedback = pipeline("python", "-m", "src.scheduler.daemon", "--run-job", "feedback")
        check.record("ground-truth feedback job", feedback.returncode == 0)

        ready_ok = wait_for(lambda: http("GET", f"{serving}/ready")[0] == 200, 60)
        status, ready = http("GET", f"{serving}/ready")
        check.record("serving ready from Redis batch view", ready_ok and ready["batch_view"]["backend"] == "redis",
                     f"{ready.get('batch_view') if isinstance(ready, dict) else ready}")

        status, pred = http("POST", f"{serving}/v1/predict/propensity", {"household_key": 7})
        check.record("prediction served from batch", status == 200 and pred.get("score_source") == "batch",
                     f"p={pred.get('unified_propensity') if isinstance(pred, dict) else pred}")
        status, _ = http("POST", f"{serving}/v1/speed/event", {"household_key": 7, "event_type": "pos_checkout"})
        check.record("event without API key rejected", status == 401, f"HTTP {status}")

        dispatched = []

        def event_applied() -> bool:
            code, res = http("POST", f"{serving}/v1/speed/event",
                             {"household_key": 7, "event_type": "pos_checkout", "sales_value": 42.0,
                              "event_id": uuid.uuid4().hex}, auth)
            dispatched.append(res.get("dispatch", {}).get("dispatched_to") if isinstance(res, dict) else code)
            _, fused = http("POST", f"{serving}/v1/predict/propensity", {"household_key": 7})
            return "intraday_purchase_dampening" in fused.get("applied_fusion_rules", [])

        applied = wait_for(event_applied, 90, interval=5)
        check.record("speed event via Kafka -> consumer -> fused score", applied and "kafka" in dispatched,
                     f"dispatched_to={sorted(set(map(str, dispatched)))}")

        status, metrics = http("GET", f"{serving}/metrics")
        check.record("/metrics exposes monitoring gauges", status == 200 and "propensity_realized_roc_auc" in metrics
                     and 'propensity_batch_view_households{variant="champion"}' in metrics)

        def targets_up() -> bool:
            _, t = http("GET", f"{prometheus}/api/v1/targets")
            jobs = {a["labels"]["job"]: a["health"] for a in t["data"]["activeTargets"]}
            return jobs.get("propensity-serving-api") == "up" and jobs.get("propensity-speed-consumer") == "up"

        check.record("Prometheus scrapes serving + consumer", wait_for(targets_up, 90))
        _, rules = http("GET", f"{prometheus}/api/v1/rules")
        n_rules = sum(len(g["rules"]) for g in rules["data"]["groups"]) if isinstance(rules, dict) else 0
        check.record("alert rules loaded", n_rules == 7, f"{n_rules} rules")
        check.record("Alertmanager ready", http("GET", f"{alertmanager}/-/ready")[0] == 200)
    finally:
        if not check.ok:
            logs = run(compose + ["logs", "--tail", "40", "serving-api", "kafka-consumer"], check=False)
            print(logs.stdout[-6000:])
        if args.keep:
            print(f"Stack left running (project {PROJECT}); env file {env_file}")
        else:
            run(compose + ["--profile", "tools", "down", "-v", "--remove-orphans"], timeout=300, check=False)

    passed = sum(ok for _, ok, _ in check.results)
    print(f"\n{'PASS' if check.ok else 'FAIL'}: {passed}/{len(check.results)} checks in {time.time() - started:.0f}s")
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
