"""
Serving load test: starts the API with uvicorn, waits for /ready, then fires concurrent
prediction traffic over all households and reports
client-side latency percentiles. Acceptance target (Phase 4): p99 < 50 ms.

Usage: python -m scripts.load_test [--requests 5000] [--concurrency 16] [--port 8765]
Rate limiting is disabled for the run (RATE_LIMIT_PER_MINUTE=0).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _call(url: str, body: dict, headers: dict) -> tuple[float, int]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    return (time.perf_counter() - start) * 1000.0, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target-p99-ms", type=float, default=50.0)
    args = parser.parse_args()

    api_key = secrets.token_hex(8)
    env = {**os.environ, "SERVING_API_KEYS": api_key, "RATE_LIMIT_PER_MINUTE": "0", "LOG_LEVEL": "WARNING"}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.serving.api:app", "--port", str(args.port), "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        deadline = time.time() + 120
        while True:
            try:
                with urllib.request.urlopen(f"{base}/ready", timeout=2) as resp:
                    ready = json.loads(resp.read())
                    break
            except urllib.error.HTTPError as e:
                if e.code == 503:
                    print("Server is up but not ready (no batch scores). Run the batch pipeline first.")
                    return 2
            except Exception:
                if server.poll() is not None:
                    print(server.stderr.read().decode(errors="replace"))
                    return 2
                if time.time() > deadline:
                    print("Server did not become ready in time")
                    return 2
                time.sleep(0.5)
        households = ready["batch_view"]["champion_households"]
        print(f"Server ready: {households:,} households in batch view ({ready['batch_view']['backend']}).")

        rng = random.Random(0)
        headers = {"Content-Type": "application/json"}
        jobs = [("predict", f"{base}/v1/predict/propensity", {"household_key": rng.randint(1, 2500)}, headers)
                for _ in range(args.requests)]

        for job in jobs[:50]:  # warm-up
            _call(*job[1:])
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda j: (j[0], *_call(*j[1:])), jobs))
        wall = time.perf_counter() - start
    finally:
        server.terminate()
        server.wait(timeout=10)

    ok = True
    print(f"\n{len(results):,} requests, concurrency {args.concurrency}, {len(results) / wall:,.0f} req/s\n")
    print(f"{'endpoint':<10}{'count':>8}{'errors':>8}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}")
    for kind in ("predict",):
        rows = [r for r in results if kind == "all" or r[0] == kind]
        lat = np.array([r[1] for r in rows])
        errors = sum(1 for r in rows if r[2] >= 400)
        p50, p95, p99 = np.percentile(lat, [50, 95, 99])
        print(f"{kind:<10}{len(rows):>8}{errors:>8}{p50:>9.1f}{p95:>9.1f}{p99:>9.1f}")
        if kind == "predict":
            ok = errors == 0 and p99 < args.target_p99_ms
    print(f"\n{'PASS' if ok else 'FAIL'}: predict p99 target < {args.target_p99_ms} ms")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
