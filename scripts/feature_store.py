"""
Feature store CLI.

    python -m scripts.feature_store status                      # snapshots in the offline history
    python -m scripts.feature_store backfill --training         # every fit/val/test cutoff of all targets
    python -m scripts.feature_store backfill --cutoffs 600,607 [--overwrite]
    python -m scripts.feature_store backfill --every 7 --start 200 --end 711
    python -m scripts.feature_store history --household 1       # PIT reads at every stored snapshot
    python -m scripts.feature_store profile --cutoff 711         # Evidently report vs the previous snapshot
    python -m scripts.feature_store profile --all [--no-html]    # every stored snapshot (trend history)
    python -m scripts.feature_store trend --feature spend_90d    # stats + drift of one feature over snapshots
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
    prof = sub.add_parser("profile")
    which = prof.add_mutually_exclusive_group(required=True)
    which.add_argument("--cutoff", type=int)
    which.add_argument("--all", action="store_true")
    prof.add_argument("--no-html", action="store_true", help="Only the JSON summary (faster, small)")
    prof.add_argument("--report-dir", default=None)
    tr = sub.add_parser("trend")
    tr.add_argument("--feature", required=True)
    tr.add_argument("--report-dir", default=None)
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
    if args.command == "profile":
        from src.feature_store.observability import profile_stored_snapshot

        days = client.snapshot_days() if args.all else [args.cutoff]
        for day in days:
            s = profile_stored_snapshot(client, day, report_dir=args.report_dir, save_html=not args.no_html)
            drift = s["drift"]
            line = f"DAY {day}"
            if drift:
                line += (f" vs {s['reference_day']}: {drift['drifted_count']} drifted {drift['drifted_columns']}, "
                         f"max out-of-range {s['max_out_of_range_share']:.2%}")
            else:
                line += ": first snapshot (statistics only)"
            print(line + (f" -> {s['html_report']}" if s.get("html_report") else ""))
        return 0
    if args.command == "trend":
        from src.feature_store.observability import trend

        table = trend(args.feature, report_dir=args.report_dir)
        if table.empty:
            print("No profiles yet; run `profile --all` first")
            return 1
        print(table.set_index("cutoff_day").to_string())
        return 0
    rows = [client.get_historical_features([args.household], d) for d in client.snapshot_days()]
    print(pd.concat(rows, ignore_index=True).set_index("cutoff_day").T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
