"""
Feature store CLI.

    python -m scripts.feature_store status                      # snapshots in the offline history
    python -m scripts.feature_store backfill --training         # every fit/val/test cutoff of all targets
    python -m scripts.feature_store backfill --cutoffs 600,607 [--overwrite]
    python -m scripts.feature_store backfill --every 7 --start 200 --end 711
    python -m scripts.feature_store history --household 1       # PIT reads at every stored snapshot
"""
from __future__ import annotations

import argparse
import logging
from typing import List

import pandas as pd

from src.config.model_config import TRAINING_CONFIGS
from src.data.loader import DataLoader
from src.feature_store.client import FeatureStoreClient
from src.features.pit_extractor import MultiDomainFeatureBuilder


def _cutoffs(args) -> List[int]:
    if args.training:
        return sorted({c for cfg in TRAINING_CONFIGS.values()
                       for c in (*cfg.fit_cutoffs, *cfg.val_cutoffs, *cfg.test_cutoffs)})
    if args.cutoffs:
        return [int(c) for c in args.cutoffs.split(",")]
    if args.every:
        return list(range(args.start, args.end + 1, args.every))
    raise SystemExit("backfill needs --training, --cutoffs or --every")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="Feature store directory (default: settings.FEATURE_STORE_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    bf = sub.add_parser("backfill")
    bf.add_argument("--training", action="store_true")
    bf.add_argument("--cutoffs", default=None)
    bf.add_argument("--every", type=int, default=None)
    bf.add_argument("--start", type=int, default=200)
    bf.add_argument("--end", type=int, default=711)
    bf.add_argument("--overwrite", action="store_true")
    bf.add_argument("--data-source", choices=["lakehouse", "csv"], default=None)
    hist = sub.add_parser("history")
    hist.add_argument("--household", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    client = FeatureStoreClient(root=args.root)
    if args.command == "status":
        days = client.snapshot_days()
        print(f"Feature store: {client.root}")
        print(f"Snapshots: {len(days)}" + (f" (DAY {days[0]} .. {days[-1]})" if days else ""))
        if days:
            print("Days:", ", ".join(map(str, days)))
        return 0
    if args.command == "backfill":
        builder = MultiDomainFeatureBuilder(DataLoader.load(args.data_source))
        written = client.backfill(builder, _cutoffs(args), overwrite=args.overwrite)
        print(f"Wrote {len(written)} snapshot(s): {written}; history now has {len(client.snapshot_days())}")
        return 0
    pd.set_option("display.width", 200)
    rows = [client.get_historical_features([args.household], d) for d in client.snapshot_days()]
    print(pd.concat(rows, ignore_index=True).set_index("cutoff_day").T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
