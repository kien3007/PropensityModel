"""Rolling backtests, model selection, calibration, and robust evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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

from .config import DEMOGRAPHIC_COLUMNS, PropensityV2Config
from .features import ID_COLUMNS


def temporal_split(
    table: pd.DataFrame,
    config: PropensityV2Config,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    snapshots = sorted(table["snapshot_day"].unique().astype(int).tolist())
    train_count = max(6, int(math.floor(len(snapshots) * config.train_fraction)))
    validation_count = max(2, int(math.floor(len(snapshots) * config.validation_fraction)))
    validation_start = train_count + 1
    validation_end = validation_start + validation_count
    test_start = validation_end + 1
    if test_start >= len(snapshots):
        raise ValueError("Insufficient snapshots after temporal embargoes")
    train_days = snapshots[:train_count]
    validation_days = snapshots[validation_start:validation_end]
    test_days = snapshots[test_start:]
    train = table["snapshot_day"].isin(train_days)
    validation = table["snapshot_day"].isin(validation_days)
    test = table["snapshot_day"].isin(test_days)
    manifest = {
        "train": {
            "snapshot_min": min(train_days),
            "snapshot_max": max(train_days),
            "snapshots": len(train_days),
            "rows": int(train.sum()),
        },
        "validation": {
            "snapshot_min": min(validation_days),
            "snapshot_max": max(validation_days),
            "snapshots": len(validation_days),
            "rows": int(validation.sum()),
        },
        "test": {
            "snapshot_min": min(test_days),
            "snapshot_max": max(test_days),
            "snapshots": len(test_days),
            "rows": int(test.sum()),
        },
        "embargo_days": [snapshots[train_count], snapshots[validation_end]],
    }
    return train, validation, test, manifest


def rolling_origin_folds(
    table: pd.DataFrame,
    primary_validation_min: int,
    folds: int = 3,
    validation_snapshots: int = 7,
) -> list[dict[str, Any]]:
    available = sorted(
        table.loc[table["snapshot_day"] < primary_validation_min, "snapshot_day"]
        .unique()
        .astype(int)
        .tolist()
    )
    required = folds * validation_snapshots + folds
    if len(available) < required + 12:
        raise ValueError("Not enough pre-validation snapshots for rolling backtests")
    records: list[dict[str, Any]] = []
    final_validation_end = len(available)
    for reverse_fold in range(folds - 1, -1, -1):
        validation_end = final_validation_end - reverse_fold * (validation_snapshots + 1)
        validation_start = validation_end - validation_snapshots
        embargo_index = validation_start - 1
        train_days = available[:embargo_index]
        validation_days = available[validation_start:validation_end]
        records.append(
            {
                "fold": len(records) + 1,
                "train_days": train_days,
                "validation_days": validation_days,
                "embargo_day": available[embargo_index],
            }
        )
    return records


def feature_contract(
    table: pd.DataFrame,
    target_column: str,
) -> tuple[list[str], list[str], list[str]]:
    categorical = [column for column in DEMOGRAPHIC_COLUMNS if column in table]
    excluded = {*ID_COLUMNS, target_column}
    numerical = [
        column
        for column in table.columns
        if column not in excluded and column not in categorical
    ]
    return numerical + categorical, numerical, categorical


def _linear_pipeline(
    numerical: list[str],
    categorical: list[str],
    random_state: int,
) -> Pipeline:
    preprocess = ColumnTransformer(
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
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    C=0.3,
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _tree_pipeline(
    numerical: list[str],
    categorical: list[str],
    random_state: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
) -> Pipeline:
    preprocess = ColumnTransformer(
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
    categorical_mask = [False] * len(numerical) + [True] * len(categorical)
    model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=180,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        categorical_features=categorical_mask,
        early_stopping=True,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def build_candidates(
    numerical: list[str],
    categorical: list[str],
    random_state: int,
) -> dict[str, Pipeline]:
    baseline_features = [
        feature
        for feature in ("recency_days", "active_days_30d", "active_days_90d", "baskets_30d")
        if feature in numerical
    ]
    return {
        "behavioral_logistic_baseline": _linear_pipeline(
            baseline_features, [], random_state
        ),
        "full_logistic_regression": _linear_pipeline(
            numerical, categorical, random_state
        ),
        "hist_gradient_boosting_shallow": _tree_pipeline(
            numerical, categorical, random_state, 15, 60, 2.0
        ),
        "hist_gradient_boosting_regularized": _tree_pipeline(
            numerical, categorical, random_state, 31, 80, 5.0
        ),
    }


def choose_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probability, np.linspace(0.03, 0.97, 95)))
    f1_values = [f1_score(target, probability >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(f1_values))])


def evaluate(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    predicted = probability >= threshold
    top_count = max(1, int(math.ceil(len(target) * top_fraction)))
    top = np.argsort(-probability)[:top_count]
    prevalence = float(np.mean(target))
    precision_at_k = float(np.mean(target[top]))
    return {
        "rows": float(len(target)),
        "positive_rate": prevalence,
        "roc_auc": float(roc_auc_score(target, probability)),
        "pr_auc": float(average_precision_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, np.clip(probability, 1e-7, 1 - 1e-7))),
        "threshold": float(threshold),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "precision_at_top_10pct": precision_at_k,
        "lift_at_top_10pct": precision_at_k / prevalence if prevalence else 0.0,
    }


def fit_calibrator(probability: np.ndarray, target: np.ndarray) -> LogisticRegression:
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    calibrator = LogisticRegression(C=1e6, max_iter=300)
    calibrator.fit(logits.reshape(-1, 1), target)
    return calibrator


def calibrate(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def top_k_metrics(target: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    records = []
    prevalence = float(np.mean(target))
    order = np.argsort(-probability)
    total_positives = int(target.sum())
    for fraction in (0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0):
        count = max(1, int(math.ceil(len(target) * fraction)))
        positives = int(target[order[:count]].sum())
        precision = positives / count
        records.append(
            {
                "population_fraction": fraction,
                "contacted_rows": count,
                "captured_buyers": positives,
                "precision": precision,
                "recall": positives / total_positives if total_positives else 0,
                "lift": precision / prevalence if prevalence else 0,
                "contacts_avoided_vs_all": len(target) - count,
            }
        )
    return pd.DataFrame(records)


def calibration_table(target: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"target": target, "probability": probability})
    frame["decile"] = pd.qcut(frame["probability"], 10, duplicates="drop")
    return (
        frame.groupby("decile", observed=True)
        .agg(
            rows=("target", "size"),
            predicted_probability=("probability", "mean"),
            observed_rate=("target", "mean"),
        )
        .reset_index(drop=True)
    )


def snapshot_metrics(predictions: pd.DataFrame, target_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for snapshot, group in predictions.groupby("snapshot_day", observed=True):
        target = group[target_column].to_numpy(dtype="int8")
        probability = group["purchase_probability"].to_numpy()
        records.append(
            {
                "snapshot_day": int(snapshot),
                "rows": len(group),
                "positive_rate": float(target.mean()),
                "pr_auc": float(average_precision_score(target, probability)),
                "roc_auc": float(roc_auc_score(target, probability)),
                "brier_score": float(brier_score_loss(target, probability)),
            }
        )
    return pd.DataFrame(records)


def segment_metrics(predictions: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frame = predictions.copy()
    frame["recency_segment"] = pd.cut(
        frame["recency_days"],
        bins=[-1, 7, 14, 30, 60, np.inf],
        labels=["0-7", "8-14", "15-30", "31-60", "61+"],
    )
    frame["demographic_segment"] = np.where(
        frame["has_demographics"].eq(1), "available", "unknown"
    )
    frame["activity_segment"] = pd.qcut(
        frame["active_days_90d"], 4, duplicates="drop"
    ).astype(str)
    records: list[dict[str, Any]] = []
    for dimension in ("recency_segment", "demographic_segment", "activity_segment"):
        for segment, group in frame.groupby(dimension, observed=True):
            target = group[target_column].to_numpy(dtype="int8")
            probability = group["purchase_probability"].to_numpy()
            record = {
                "dimension": dimension,
                "segment": str(segment),
                "rows": len(group),
                "positive_rate": float(target.mean()),
                "mean_probability": float(probability.mean()),
                "brier_score": float(brier_score_loss(target, probability)),
                "pr_auc": np.nan,
                "roc_auc": np.nan,
            }
            if len(np.unique(target)) == 2:
                record["pr_auc"] = float(average_precision_score(target, probability))
                record["roc_auc"] = float(roc_auc_score(target, probability))
            records.append(record)
    return pd.DataFrame(records)


def bootstrap_intervals(
    predictions: pd.DataFrame,
    target_column: str,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    grouped = {
        household: group.index.to_numpy()
        for household, group in predictions.groupby("household_key", observed=True)
    }
    households = np.asarray(list(grouped))
    records: list[dict[str, float]] = []
    for _ in range(repeats):
        sampled = rng.choice(households, size=len(households), replace=True)
        indices = np.concatenate([grouped[household] for household in sampled])
        sample = predictions.loc[indices]
        target = sample[target_column].to_numpy(dtype="int8")
        probability = sample["purchase_probability"].to_numpy()
        top_count = max(1, int(math.ceil(len(target) * 0.10)))
        top = np.argsort(-probability)[:top_count]
        prevalence = float(target.mean())
        records.append(
            {
                "pr_auc": average_precision_score(target, probability),
                "roc_auc": roc_auc_score(target, probability),
                "brier_score": brier_score_loss(target, probability),
                "lift_at_top_10pct": float(target[top].mean()) / prevalence,
            }
        )
    samples = pd.DataFrame(records)
    intervals = []
    for metric in samples.columns:
        intervals.append(
            {
                "metric": metric,
                "estimate_mean": float(samples[metric].mean()),
                "ci_lower_95": float(samples[metric].quantile(0.025)),
                "ci_upper_95": float(samples[metric].quantile(0.975)),
            }
        )
    return pd.DataFrame(intervals)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    edges = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = pd.cut(
        reference, edges, include_lowest=True
    ).value_counts(normalize=True)
    current_share = pd.cut(current, edges, include_lowest=True).value_counts(normalize=True)
    reference_share = reference_share.clip(lower=1e-6)
    current_share = current_share.reindex(
        reference_share.index, fill_value=1e-6
    ).clip(lower=1e-6)
    return float(
        (
            (current_share - reference_share)
            * np.log(current_share / reference_share)
        ).sum()
    )


def feature_family(feature: str) -> str:
    if feature in DEMOGRAPHIC_COLUMNS or feature == "has_demographics":
        return "demographics"
    if any(token in feature for token in ("discount", "coupon")):
        return "promotion_dependency"
    if any(
        token in feature
        for token in ("grocery", "fresh", "drug_gm", "fuel", "other_spend")
    ):
        return "category_mix"
    if "private_label" in feature:
        return "brand_mix"
    if any(token in feature for token in ("morning", "afternoon", "evening")):
        return "shopping_daypart"
    if "diversity" in feature:
        return "assortment_and_store_diversity"
    if any(token in feature for token in ("recency", "active_days", "gap", "cadence")):
        return "purchase_cadence"
    if any(token in feature for token in ("basket", "line_items")):
        return "basket_frequency_and_size"
    if any(token in feature for token in ("spend", "monetary")):
        return "monetary_value"
    if any(token in feature for token in ("tenure", "lifetime")):
        return "lifecycle"
    return "other_behavior"


def grouped_permutation_importance(
    model: Pipeline,
    features: list[str],
    x_sample: pd.DataFrame,
    target: np.ndarray,
    random_state: int,
) -> pd.DataFrame:
    baseline = average_precision_score(target, model.predict_proba(x_sample)[:, 1])
    rng = np.random.default_rng(random_state)
    families: dict[str, list[str]] = {}
    for feature in features:
        families.setdefault(feature_family(feature), []).append(feature)
    records = []
    for family, columns in families.items():
        shuffled = x_sample.copy()
        permutation = rng.permutation(len(shuffled))
        for column in columns:
            shuffled[column] = x_sample[column].iloc[permutation].to_numpy()
        score = average_precision_score(target, model.predict_proba(shuffled)[:, 1])
        records.append(
            {
                "feature_family": family,
                "features_in_family": len(columns),
                "baseline_pr_auc": baseline,
                "permuted_pr_auc": score,
                "importance_pr_auc_drop": baseline - score,
            }
        )
    return pd.DataFrame(records).sort_values("importance_pr_auc_drop", ascending=False)


def drift_classification(feature: str, psi: float) -> tuple[str, str, str]:
    structural = feature == "tenure_days" or feature.startswith("lifetime_")
    if structural:
        return (
            "expected_structural",
            "informational",
            "Cumulative features naturally increase with later snapshots; monitor "
            "performance, not a fixed PSI alert.",
        )
    if psi < 0.10:
        return "rolling_behavior", "low", "No action."
    if psi < 0.25:
        return "rolling_behavior", "moderate", "Review distribution and segment performance."
    return "rolling_behavior", "high", "Investigate upstream changes before scoring."


def train_and_evaluate(
    table: pd.DataFrame,
    config: PropensityV2Config,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_column = config.target_column
    features, numerical, categorical = feature_contract(table, target_column)
    train_mask, validation_mask, test_mask, split_manifest = temporal_split(table, config)
    folds = rolling_origin_folds(
        table, split_manifest["validation"]["snapshot_min"]
    )
    candidate_results: list[dict[str, Any]] = []
    for fold in folds:
        fold_train = table["snapshot_day"].isin(fold["train_days"])
        fold_validation = table["snapshot_day"].isin(fold["validation_days"])
        x_train = table.loc[fold_train, features]
        y_train = table.loc[fold_train, target_column].to_numpy(dtype="int8")
        x_validation = table.loc[fold_validation, features]
        y_validation = table.loc[fold_validation, target_column].to_numpy(dtype="int8")
        for name, candidate in build_candidates(
            numerical, categorical, config.random_state
        ).items():
            candidate.fit(x_train, y_train)
            probability = candidate.predict_proba(x_validation)[:, 1]
            result = evaluate(
                y_validation,
                probability,
                choose_threshold(y_validation, probability),
                config.top_k_fraction,
            )
            candidate_results.append(
                {
                    "fold": fold["fold"],
                    "model": name,
                    "train_snapshot_max": max(fold["train_days"]),
                    "embargo_day": fold["embargo_day"],
                    "validation_snapshot_min": min(fold["validation_days"]),
                    "validation_snapshot_max": max(fold["validation_days"]),
                    **result,
                }
            )
    backtest = pd.DataFrame(candidate_results)
    backtest.to_csv(output_dir / "rolling_backtest_metrics.csv", index=False)
    summary = (
        backtest.groupby("model", observed=True)
        .agg(
            folds=("fold", "nunique"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_lift_at_10pct=("lift_at_top_10pct", "mean"),
        )
        .reset_index()
        .sort_values(["mean_pr_auc", "mean_brier_score"], ascending=[False, True])
    )
    summary.to_csv(output_dir / "model_selection_summary.csv", index=False)
    selected_name = str(summary.iloc[0]["model"])

    candidates = build_candidates(numerical, categorical, config.random_state)
    selected = candidates[selected_name]
    x_train = table.loc[train_mask, features]
    y_train = table.loc[train_mask, target_column].to_numpy(dtype="int8")
    x_validation = table.loc[validation_mask, features]
    y_validation = table.loc[validation_mask, target_column].to_numpy(dtype="int8")
    selected.fit(x_train, y_train)
    validation_raw = selected.predict_proba(x_validation)[:, 1]
    calibrator = fit_calibrator(validation_raw, y_validation)
    validation_probability = calibrate(calibrator, validation_raw)
    threshold = choose_threshold(y_validation, validation_probability)
    validation_metrics = evaluate(
        y_validation, validation_probability, threshold, config.top_k_fraction
    )

    x_test = table.loc[test_mask, features]
    y_test = table.loc[test_mask, target_column].to_numpy(dtype="int8")
    test_probability = calibrate(calibrator, selected.predict_proba(x_test)[:, 1])
    test_metrics = evaluate(y_test, test_probability, threshold, config.top_k_fraction)

    predictions = table.loc[
        test_mask,
        ID_COLUMNS
        + [
            target_column,
            "recency_days",
            "active_days_90d",
            "has_demographics",
        ],
    ].copy()
    predictions["purchase_probability"] = test_probability
    predictions["prediction"] = (test_probability >= threshold).astype("int8")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    top_k_metrics(y_test, test_probability).to_csv(
        output_dir / "top_k_business_metrics.csv", index=False
    )
    calibration_table(y_test, test_probability).to_csv(
        output_dir / "test_calibration.csv", index=False
    )
    snapshot_metrics(predictions, target_column).to_csv(
        output_dir / "test_metrics_by_snapshot.csv", index=False
    )
    segment_metrics(predictions, target_column).to_csv(
        output_dir / "test_metrics_by_segment.csv", index=False
    )
    bootstrap_intervals(
        predictions, target_column, config.bootstrap_repeats, config.random_state
    ).to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)

    sample_size = min(5000, len(x_test))
    sample_positions = np.linspace(0, len(x_test) - 1, sample_size, dtype=int)
    importance = permutation_importance(
        selected,
        x_test.iloc[sample_positions],
        y_test[sample_positions],
        scoring="average_precision",
        n_repeats=1,
        random_state=config.random_state,
        n_jobs=1,
    )
    pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    grouped_permutation_importance(
        selected,
        features,
        x_test.iloc[sample_positions].copy(),
        y_test[sample_positions],
        config.random_state,
    ).to_csv(output_dir / "grouped_feature_importance.csv", index=False)

    drift_records = []
    for feature in numerical:
        psi = population_stability_index(
            table.loc[train_mask, feature], table.loc[test_mask, feature]
        )
        drift_type, severity, action = drift_classification(feature, psi)
        drift_records.append(
            {
                "feature": feature,
                "psi_train_to_test": psi,
                "drift_type": drift_type,
                "severity": severity,
                "monitoring_action": action,
            }
        )
    pd.DataFrame(drift_records).sort_values(
        "psi_train_to_test", ascending=False
    ).to_csv(output_dir / "feature_drift_psi.csv", index=False)

    bundle = {
        "model": selected,
        "calibrator": calibrator,
        "threshold": threshold,
        "features": features,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "selected_model": selected_name,
        "config": asdict(config),
    }
    joblib.dump(bundle, output_dir / "propensity_v2_model.joblib")
    return {
        "selected_model": selected_name,
        "split_manifest": split_manifest,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_selection": summary.to_dict(orient="records"),
        "feature_count": len(features),
    }
