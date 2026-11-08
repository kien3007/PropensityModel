"""Command-line training and batch-scoring entry point for propensity V2."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import PropensityV2Config
from .data import (
    build_daily_features,
    enrich_transactions,
    load_demographics,
    load_transactions,
)
from .features import ID_COLUMNS, build_modeling_table
from .modeling import calibrate, train_and_evaluate


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_native(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def prepare(raw_dir: Path, config: PropensityV2Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    transactions = load_transactions(raw_dir)
    enriched = enrich_transactions(transactions, raw_dir)
    daily, basket = build_daily_features(enriched)
    households = np.sort(daily["household_key"].unique())
    demographics = load_demographics(raw_dir, households)
    table, profile = build_modeling_table(daily, demographics, config)
    profile.update(
        {
            "transaction_rows": len(transactions),
            "basket_rows": len(basket),
            "product_match_rate": float(enriched["DEPARTMENT"].notna().mean()),
            "demographic_household_coverage": float(
                demographics["has_demographics"].mean()
            ),
            "quantity_excluded": True,
            "quantity_exclusion_reason": (
                "KIOSK-GAS and MISC SALES use a scaled quantity convention that is not "
                "comparable to grocery item counts"
            ),
        }
    )
    return table, profile


def train(raw_dir: Path, output_dir: Path, config: PropensityV2Config) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table, profile = prepare(raw_dir, config)
    if table.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate household-snapshot keys")
    if table.drop(columns=[config.target_column]).isna().any().any():
        raise ValueError("Unexpected missing values in the modeling table")
    model_results = train_and_evaluate(table, config, output_dir)
    table.to_parquet(output_dir / "modeling_table_v2.parquet", index=False)
    metadata = {
        "project": "Dunnhumby household purchase propensity V2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_definition": (
            f"P(at least one basket in (snapshot, snapshot + {config.horizon_days}] "
            "given information available at snapshot)"
        ),
        "config": asdict(config),
        "data_profile": profile,
        **model_results,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    write_json(output_dir / "metrics_v2.json", metadata)
    return metadata


def score(raw_dir: Path, model_path: Path, output_path: Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    config = PropensityV2Config(**bundle["config"])
    transactions = load_transactions(raw_dir)
    enriched = enrich_transactions(transactions, raw_dir)
    daily, _ = build_daily_features(enriched)
    households = np.sort(daily["household_key"].unique())
    demographics = load_demographics(raw_dir, households)
    latest_day = int(daily["DAY"].max())
    table, _ = build_modeling_table(
        daily,
        demographics,
        config,
        snapshots=[latest_day],
        include_target=False,
    )
    raw_probability = bundle["model"].predict_proba(table[bundle["features"]])[:, 1]
    probability = calibrate(bundle["calibrator"], raw_probability)
    scores = table[ID_COLUMNS].copy()
    scores["purchase_probability_next_7d"] = probability
    scores["high_propensity_flag"] = (
        probability >= float(bundle["threshold"])
    ).astype("int8")
    scores["rank"] = scores["purchase_probability_next_7d"].rank(
        method="first", ascending=False
    ).astype(int)
    scores["percentile"] = 1 - (scores["rank"] - 1) / len(scores)
    scores = scores.sort_values("rank")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--raw-dir", type=Path, default=Path("sample_data/dunnhumby")
    )
    train_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/propensity_v2/model")
    )
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument(
        "--raw-dir", type=Path, default=Path("sample_data/dunnhumby")
    )
    score_parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/propensity_v2/model/propensity_v2_model.joblib"),
    )
    score_parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/propensity_v2/model/latest_scores_v2.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        result = train(args.raw_dir, args.output_dir, PropensityV2Config())
        print(json.dumps(_native(result), indent=2, ensure_ascii=False))
    else:
        scores = score(args.raw_dir, args.model_path, args.output_path)
        print(f"Scored {len(scores):,} households -> {args.output_path}")


if __name__ == "__main__":
    main()
