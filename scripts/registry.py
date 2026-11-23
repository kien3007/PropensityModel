"""
Champion registry CLI.

    python -m scripts.registry [--target purchase|lapse] list
    python -m scripts.registry [--target purchase|lapse] rollback [--to VERSION] [--reason TEXT]
"""
from __future__ import annotations

import logging

from src.config.model_config import TRAINING_CONFIGS
from src.training.model_registry import ModelRegistry, registry_dir_for


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Champion model registry")
    parser.add_argument("--target", choices=sorted(TRAINING_CONFIGS), default="purchase")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List archived versions (* = active champion)")
    rb = sub.add_parser("rollback", help="Re-activate a previous champion")
    rb.add_argument("--to", dest="to_version", default=None, help="Version id (default: the previous one)")
    rb.add_argument("--reason", default="manual rollback")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    registry = ModelRegistry(registry_dir_for(TRAINING_CONFIGS[args.target]))
    if args.command == "list":
        current = registry.current_version()
        for v in registry.list_versions():
            print(("* " if v == current else "  ") + v)
        if current is None:
            print("(active champion is not archived yet; it is archived automatically on the next promotion)")
        return 0
    try:
        version = registry.rollback(args.to_version, args.reason)
    except ValueError as exc:
        print(f"Rollback refused: {exc}")
        return 1
    print(f"Champion is now version {version}. POST /v1/models/reload to hot-swap the serving layer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
