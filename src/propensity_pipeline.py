"""Point-in-time correct household purchase propensity pipeline.

The prediction unit is one (household, snapshot_day).  Features include events
on or before snapshot_day.  The target is one when at least one basket occurs
in (snapshot_day, snapshot_day + horizon_days].
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings(
    "ignore",
    message=r"(?s).*physical cores.*",
    category=UserWarning,
)


ID_COLUMNS = ["household_key", "snapshot_day"]
TARGET_COLUMN = "target_purchase_next_7d"
WINDOWS = (7, 30, 90, 180)
DAILY_METRICS = ("spend", "units", "baskets", "discount", "active_days")
DEMOGRAPHIC_COLUMNS = (
    "classification_1",
    "classification_2",
    "classification_3",
    "HOMEOWNER_DESC",
    "classification_5",
    "classification_4",
    "KID_CATEGORY_DESC",
)


@dataclass(frozen=True)
class PropensityConfig:
    """Reproducible temporal and modeling configuration."""

    horizon_days: int = 7
    snapshot_step_days: int = 7
    minimum_history_days: int = 180
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    random_state: int = 42
    top_k_fraction: float = 0.10


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_builtin(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_transactions(raw_dir: Path) -> pd.DataFrame:
    """Load only columns needed for basket-level behavioral features."""

    transaction_path = raw_dir / "transaction_data.csv"
    required = {
        "household_key": "int32",
        "BASKET_ID": "int64",
        "DAY": "int16",
        "PRODUCT_ID": "int64",
        "QUANTITY": "int32",
        "SALES_VALUE": "float32",
        "RETAIL_DISC": "float32",
        "COUPON_DISC": "float32",
        "COUPON_MATCH_DISC": "float32",
    }
    if not transaction_path.exists():
        raise FileNotFoundError(f"Missing transaction file: {transaction_path}")
    transactions = pd.read_csv(
        transaction_path,
        usecols=list(required),
        dtype=required,
    )
    if transactions.empty:
        raise ValueError("transaction_data.csv contains no rows")
    return transactions


def build_daily_behavior(transactions: pd.DataFrame) -> pd.DataFrame:
    """Collapse item rows to household-day behavior without basket duplication."""

    frame = transactions.copy()
    frame["discount"] = -(
        frame["RETAIL_DISC"]
        + frame["COUPON_DISC"]
        + frame["COUPON_MATCH_DISC"]
    )
    frame["discount"] = frame["discount"].clip(lower=0)
    daily = (
        frame.groupby(["household_key", "DAY"], sort=True, observed=True)
        .agg(
            spend=("SALES_VALUE", "sum"),
            units=("QUANTITY", "sum"),
            baskets=("BASKET_ID", "nunique"),
            discount=("discount", "sum"),
        )
        .reset_index()
    )
    daily["active_days"] = 1.0
    for column in DAILY_METRICS:
        daily[column] = daily[column].astype("float32")
    return daily


def load_demographics(raw_dir: Path, households: Iterable[int]) -> pd.DataFrame:
    path = raw_dir / "hh_demographic.csv"
    base = pd.DataFrame({"household_key": np.asarray(list(households), dtype="int32")})
    if not path.exists():
        for column in DEMOGRAPHIC_COLUMNS:
            base[column] = "Unknown"
        return base
    demographics = pd.read_csv(path, dtype={"household_key": "int32"})
    available = [column for column in DEMOGRAPHIC_COLUMNS if column in demographics]
    demographics = demographics[["household_key", *available]].drop_duplicates(
        "household_key", keep="last"
    )
    result = base.merge(demographics, on="household_key", how="left")
    for column in DEMOGRAPHIC_COLUMNS:
        if column not in result:
            result[column] = "Unknown"
        result[column] = result[column].fillna("Unknown").astype(str)
    return result


def _dense_metric(
    daily: pd.DataFrame,
    households: np.ndarray,
    maximum_day: int,
    metric: str,
) -> np.ndarray:
    matrix = (
        daily.pivot(index="household_key", columns="DAY", values=metric)
        .reindex(index=households, columns=range(maximum_day + 1), fill_value=0)
        .fillna(0)
        .to_numpy(dtype="float32")
    )
    return matrix


def _window_sum(cumulative: np.ndarray, snapshot: int, window: int) -> np.ndarray:
    start = max(snapshot - window, 0)
    return cumulative[:, snapshot] - cumulative[:, start]


def construct_modeling_table(
    daily: pd.DataFrame,
    demographics: pd.DataFrame,
    config: PropensityConfig,
    snapshots: list[int] | None = None,
    include_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create household snapshots using dense cumulative arrays.

    A feature at day s uses days (s-window, s], while the label uses (s, s+h].
    """

    households = np.sort(daily["household_key"].unique().astype("int32"))
    minimum_day = int(daily["DAY"].min())
    maximum_day = int(daily["DAY"].max())
    last_eligible_snapshot = maximum_day - config.horizon_days
    if snapshots is None:
        first_snapshot = minimum_day + config.minimum_history_days - 1
        snapshots = list(
            range(
                first_snapshot,
                last_eligible_snapshot + 1,
                config.snapshot_step_days,
            )
        )
    if not snapshots:
        raise ValueError("No eligible snapshot days for the configured history and horizon")
    if min(snapshots) < 0 or max(snapshots) > maximum_day:
        raise ValueError("Snapshot day is outside transaction coverage")
    if include_target and max(snapshots) + config.horizon_days > maximum_day:
        raise ValueError("Target horizon is not mature for the latest snapshot")

    dense: dict[str, np.ndarray] = {}
    cumulative: dict[str, np.ndarray] = {}
    for metric in DAILY_METRICS:
        dense[metric] = _dense_metric(daily, households, maximum_day, metric)
        cumulative[metric] = np.cumsum(dense[metric], axis=1, dtype="float64")

    purchase_mask = dense["baskets"] > 0
    day_axis = np.arange(maximum_day + 1, dtype="int16")[None, :]
    last_purchase = np.maximum.accumulate(np.where(purchase_mask, day_axis, -1), axis=1)
    first_purchase = np.where(purchase_mask, day_axis, maximum_day + 1).min(axis=1)

    frames: list[pd.DataFrame] = []
    for snapshot in snapshots:
        payload: dict[str, Any] = {
            "household_key": households,
            "snapshot_day": np.full(len(households), snapshot, dtype="int16"),
            "recency_days": np.where(
                last_purchase[:, snapshot] >= 0,
                snapshot - last_purchase[:, snapshot],
                config.minimum_history_days,
            ).astype("float32"),
            "tenure_days": np.maximum(snapshot - first_purchase, 0).astype("float32"),
        }
        for window in WINDOWS:
            for metric in DAILY_METRICS:
                payload[f"{metric}_{window}d"] = _window_sum(
                    cumulative[metric], snapshot, window
                ).astype("float32")
            baskets = payload[f"baskets_{window}d"]
            spend = payload[f"spend_{window}d"]
            discount = payload[f"discount_{window}d"]
            payload[f"avg_basket_value_{window}d"] = np.divide(
                spend,
                baskets,
                out=np.zeros_like(spend),
                where=baskets > 0,
            )
            payload[f"discount_rate_{window}d"] = np.divide(
                discount,
                spend + discount,
                out=np.zeros_like(discount),
                where=(spend + discount) > 0,
            )

        prior_30_start = max(snapshot - 60, 0)
        prior_30_end = max(snapshot - 30, 0)
        for metric in ("spend", "baskets", "active_days"):
            prior = (
                cumulative[metric][:, prior_30_end]
                - cumulative[metric][:, prior_30_start]
            ).astype("float32")
            recent = payload[f"{metric}_30d"]
            payload[f"{metric}_trend_30d"] = np.log1p(recent) - np.log1p(prior)

        lifetime_baskets = cumulative["baskets"][:, snapshot].astype("float32")
        lifetime_spend = cumulative["spend"][:, snapshot].astype("float32")
        payload["lifetime_baskets"] = lifetime_baskets
        payload["lifetime_spend"] = lifetime_spend
        payload["lifetime_avg_basket_value"] = np.divide(
            lifetime_spend,
            lifetime_baskets,
            out=np.zeros_like(lifetime_spend),
            where=lifetime_baskets > 0,
        )
        if include_target:
            future_baskets = (
                cumulative["baskets"][:, snapshot + config.horizon_days]
                - cumulative["baskets"][:, snapshot]
            )
            payload[TARGET_COLUMN] = (future_baskets > 0).astype("int8")
        known_household = first_purchase <= snapshot
        frames.append(pd.DataFrame(payload).loc[known_household].copy())

    table = pd.concat(frames, ignore_index=True)
    table = table.merge(demographics, on="household_key", how="left", validate="many_to_one")
    for column in DEMOGRAPHIC_COLUMNS:
        table[column] = table[column].fillna("Unknown").astype(str)

    profile = {
        "raw_day_min": minimum_day,
        "raw_day_max": maximum_day,
        "households": len(households),
        "daily_household_rows": len(daily),
        "snapshot_count": len(snapshots),
        "snapshot_min": min(snapshots),
        "snapshot_max": max(snapshots),
        "modeling_rows": len(table),
        "target_rate": float(table[TARGET_COLUMN].mean()) if include_target else None,
        "demographic_coverage_rate": float(
            (table["classification_1"] != "Unknown").mean()
        ),
    }
    return table, profile


def temporal_split(
    table: pd.DataFrame,
    config: PropensityConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    """Chronological split with one full horizon embargo between partitions."""

    snapshots = sorted(int(value) for value in table["snapshot_day"].unique())
    if len(snapshots) < 12:
        raise ValueError("At least 12 snapshots are required for temporal evaluation")
    train_count = max(6, int(math.floor(len(snapshots) * config.train_fraction)))
    validation_count = max(
        2,
        int(math.floor(len(snapshots) * config.validation_fraction)),
    )
    validation_start = train_count + 1
    validation_end = validation_start + validation_count
    test_start = validation_end + 1
    if test_start >= len(snapshots):
        raise ValueError("Not enough snapshots after applying temporal embargoes")

    train_days = snapshots[:train_count]
    validation_days = snapshots[validation_start:validation_end]
    test_days = snapshots[test_start:]
    train_mask = table["snapshot_day"].isin(train_days)
    validation_mask = table["snapshot_day"].isin(validation_days)
    test_mask = table["snapshot_day"].isin(test_days)
    manifest = {
        "train": {
            "snapshot_min": min(train_days),
            "snapshot_max": max(train_days),
            "rows": int(train_mask.sum()),
        },
        "validation": {
            "snapshot_min": min(validation_days),
            "snapshot_max": max(validation_days),
            "rows": int(validation_mask.sum()),
        },
        "test": {
            "snapshot_min": min(test_days),
            "snapshot_max": max(test_days),
            "rows": int(test_mask.sum()),
        },
        "embargo_snapshot_days": [
            snapshots[train_count],
            snapshots[validation_end],
        ],
        "horizon_days": config.horizon_days,
    }
    return train_mask, validation_mask, test_mask, manifest


def feature_columns(table: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = {*ID_COLUMNS, TARGET_COLUMN}
    categorical = [column for column in DEMOGRAPHIC_COLUMNS if column in table]
    numerical = [
        column
        for column in table.columns
        if column not in excluded and column not in categorical
    ]
    return numerical + categorical, numerical, categorical


def build_candidates(
    numerical: list[str],
    categorical: list[str],
    random_state: int,
) -> dict[str, Pipeline]:
    linear_preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                categorical,
            ),
        ]
    )
    tree_preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numerical),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                categorical,
            ),
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", linear_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        max_iter=500,
                        class_weight=None,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.07,
                        max_iter=180,
                        max_leaf_nodes=31,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def fit_probability_calibrator(probability: np.ndarray, target: np.ndarray) -> LogisticRegression:
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    calibrator = LogisticRegression(C=1e6, max_iter=200)
    calibrator.fit(logits.reshape(-1, 1), target)
    return calibrator


def apply_probability_calibrator(
    calibrator: LogisticRegression,
    probability: np.ndarray,
) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def choose_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probability, np.linspace(0.05, 0.95, 91)))
    scores = [f1_score(target, probability >= value) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate_predictions(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    top_k_fraction: float,
) -> dict[str, float]:
    prediction = probability >= threshold
    top_count = max(1, int(math.ceil(len(target) * top_k_fraction)))
    top_index = np.argsort(-probability)[:top_count]
    prevalence = float(np.mean(target))
    precision_at_k = float(np.mean(target[top_index]))
    return {
        "rows": float(len(target)),
        "positive_rate": prevalence,
        "roc_auc": float(roc_auc_score(target, probability)),
        "pr_auc": float(average_precision_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "threshold": float(threshold),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "precision_at_top_10pct": precision_at_k,
        "lift_at_top_10pct": precision_at_k / prevalence if prevalence > 0 else 0.0,
    }


def calibration_table(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"target": target, "probability": probability})
    frame["bin"] = pd.qcut(frame["probability"], q=bins, duplicates="drop")
    result = (
        frame.groupby("bin", observed=True)
        .agg(
            rows=("target", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_purchase_rate=("target", "mean"),
        )
        .reset_index(drop=True)
    )
    return result


def data_quality_checks(
    transactions: pd.DataFrame,
    table: pd.DataFrame,
    config: PropensityConfig,
) -> dict[str, Any]:
    duplicated_keys = int(table.duplicated(ID_COLUMNS).sum())
    checks = {
        "transaction_rows": len(transactions),
        "transaction_null_household": int(transactions["household_key"].isna().sum()),
        "transaction_null_day": int(transactions["DAY"].isna().sum()),
        "transaction_duplicate_item_rows": int(transactions.duplicated().sum()),
        "modeling_duplicate_keys": duplicated_keys,
        "modeling_missing_feature_cells": int(
            table.drop(columns=[TARGET_COLUMN]).isna().sum().sum()
        ),
        "target_values": sorted(table[TARGET_COLUMN].unique().tolist()),
        "target_definition": (
            "at least one basket in (snapshot_day, snapshot_day + "
            f"{config.horizon_days}]"
        ),
        "features_end_at_snapshot": True,
    }
    if duplicated_keys:
        raise ValueError("Duplicate household-snapshot keys found")
    if not set(table[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target is not binary")
    return checks


def train_pipeline(
    raw_dir: Path,
    output_dir: Path,
    config: PropensityConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transactions = load_transactions(raw_dir)
    daily = build_daily_behavior(transactions)
    households = np.sort(daily["household_key"].unique())
    demographics = load_demographics(raw_dir, households)
    table, profile = construct_modeling_table(daily, demographics, config)
    quality = data_quality_checks(transactions, table, config)
    train_mask, validation_mask, test_mask, split_manifest = temporal_split(table, config)
    features, numerical, categorical = feature_columns(table)
    target = table[TARGET_COLUMN].to_numpy(dtype="int8")

    x_train = table.loc[train_mask, features]
    y_train = target[train_mask.to_numpy()]
    x_validation = table.loc[validation_mask, features]
    y_validation = target[validation_mask.to_numpy()]
    x_test = table.loc[test_mask, features]
    y_test = target[test_mask.to_numpy()]

    candidates = build_candidates(numerical, categorical, config.random_state)
    validation_results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        probability = candidate.predict_proba(x_validation)[:, 1]
        validation_results[name] = evaluate_predictions(
            y_validation,
            probability,
            choose_threshold(y_validation, probability),
            config.top_k_fraction,
        )
        fitted[name] = candidate

    winner = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["pr_auc"],
            -validation_results[name]["brier_score"],
        ),
    )
    selected_model = fitted[winner]
    raw_validation_probability = selected_model.predict_proba(x_validation)[:, 1]
    calibrator = fit_probability_calibrator(raw_validation_probability, y_validation)
    calibrated_validation_probability = apply_probability_calibrator(
        calibrator, raw_validation_probability
    )
    threshold = choose_threshold(y_validation, calibrated_validation_probability)

    raw_test_probability = selected_model.predict_proba(x_test)[:, 1]
    test_probability = apply_probability_calibrator(calibrator, raw_test_probability)
    test_metrics = evaluate_predictions(
        y_test,
        test_probability,
        threshold,
        config.top_k_fraction,
    )

    test_predictions = table.loc[test_mask, ID_COLUMNS + [TARGET_COLUMN]].copy()
    test_predictions["purchase_probability"] = test_probability
    test_predictions["prediction"] = (test_probability >= threshold).astype("int8")
    test_predictions = test_predictions.sort_values(
        ["snapshot_day", "purchase_probability"], ascending=[True, False]
    )
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    calibration = calibration_table(y_test, test_probability)
    calibration.to_csv(output_dir / "test_calibration.csv", index=False)

    sample_size = min(20000, len(x_test))
    sampled_positions = np.linspace(0, len(x_test) - 1, sample_size, dtype=int)
    importance = permutation_importance(
        selected_model,
        x_test.iloc[sampled_positions],
        y_test[sampled_positions],
        scoring="average_precision",
        n_repeats=3,
        random_state=config.random_state,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.to_csv(output_dir / "feature_importance.csv", index=False)

    bundle = {
        "model": selected_model,
        "calibrator": calibrator,
        "threshold": threshold,
        "features": features,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "config": asdict(config),
        "selected_model": winner,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, output_dir / "propensity_model.joblib")

    metadata = {
        "config": asdict(config),
        "data_profile": profile,
        "data_quality": quality,
        "split_manifest": split_manifest,
        "validation_metrics": validation_results,
        "selected_model": winner,
        "test_metrics": test_metrics,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    write_json(output_dir / "metrics.json", metadata)

    try:
        table.to_parquet(output_dir / "modeling_table.parquet", index=False)
    except ImportError:
        table.to_csv(output_dir / "modeling_table.csv.gz", index=False)
    return metadata


def score_latest_snapshot(raw_dir: Path, model_path: Path, output_path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bundle = joblib.load(model_path)
    config = PropensityConfig(**bundle["config"])
    transactions = load_transactions(raw_dir)
    daily = build_daily_behavior(transactions)
    households = np.sort(daily["household_key"].unique())
    demographics = load_demographics(raw_dir, households)
    latest_day = int(daily["DAY"].max())
    table, _ = construct_modeling_table(
        daily,
        demographics,
        config,
        snapshots=[latest_day],
        include_target=False,
    )
    raw_probability = bundle["model"].predict_proba(table[bundle["features"]])[:, 1]
    probability = apply_probability_calibrator(bundle["calibrator"], raw_probability)
    result = table[ID_COLUMNS].copy()
    result["purchase_probability_next_7d"] = probability
    result["high_propensity_flag"] = (
        probability >= float(bundle["threshold"])
    ).astype("int8")
    result = result.sort_values("purchase_probability_next_7d", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="build snapshots, train, and evaluate")
    train.add_argument("--raw-dir", type=Path, default=Path("sample_data/dunnhumby"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/propensity"))
    score = subparsers.add_parser("score", help="score all households at latest raw day")
    score.add_argument("--raw-dir", type=Path, default=Path("sample_data/dunnhumby"))
    score.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/propensity/propensity_model.joblib"),
    )
    score.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/propensity/latest_scores.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        result = train_pipeline(args.raw_dir, args.output_dir, PropensityConfig())
        print(json.dumps(_to_builtin(result), indent=2, ensure_ascii=False))
    else:
        scored = score_latest_snapshot(args.raw_dir, args.model_path, args.output_path)
        print(f"Scored {len(scored):,} households -> {args.output_path}")


if __name__ == "__main__":
    main()
