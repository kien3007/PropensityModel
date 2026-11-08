"""Point-in-time snapshot and feature engineering for propensity V2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ADDITIVE_DAILY_METRICS,
    DEMOGRAPHIC_COLUMNS,
    WINDOWS,
    PropensityV2Config,
)


ID_COLUMNS = ["household_key", "snapshot_day"]


def _dense_metric(
    daily: pd.DataFrame,
    households: np.ndarray,
    maximum_day: int,
    metric: str,
) -> np.ndarray:
    return (
        daily.pivot(index="household_key", columns="DAY", values=metric)
        .reindex(index=households, columns=range(maximum_day + 1), fill_value=0)
        .fillna(0)
        .to_numpy(dtype="float32")
    )


def _window(cumulative: np.ndarray, snapshot: int, days: int) -> np.ndarray:
    return cumulative[:, snapshot] - cumulative[:, max(snapshot - days, 0)]


def _purchase_state(purchase_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    households, days = purchase_mask.shape
    last = np.full((households, days), -1, dtype="int16")
    previous = np.full((households, days), -1, dtype="int16")
    last_state = np.full(households, -1, dtype="int16")
    previous_state = np.full(households, -1, dtype="int16")
    for day in range(days):
        purchased = purchase_mask[:, day]
        previous_state[purchased] = last_state[purchased]
        last_state[purchased] = day
        last[:, day] = last_state
        previous[:, day] = previous_state
    return last, previous


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype="float32"),
        where=denominator > 0,
    ).astype("float32")


def build_modeling_table(
    daily: pd.DataFrame,
    demographics: pd.DataFrame,
    config: PropensityV2Config,
    snapshots: list[int] | None = None,
    include_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    households = np.sort(daily["household_key"].unique().astype("int32"))
    minimum_day = int(daily["DAY"].min())
    maximum_day = int(daily["DAY"].max())
    if snapshots is None:
        first_snapshot = minimum_day + config.minimum_history_days - 1
        last_snapshot = maximum_day - config.horizon_days
        snapshots = list(
            range(first_snapshot, last_snapshot + 1, config.snapshot_step_days)
        )
    if not snapshots:
        raise ValueError("No eligible snapshot days")
    if include_target and max(snapshots) + config.horizon_days > maximum_day:
        raise ValueError("Latest target has not matured")

    dense = {
        metric: _dense_metric(daily, households, maximum_day, metric)
        for metric in ADDITIVE_DAILY_METRICS
    }
    cumulative = {
        metric: np.cumsum(values, axis=1, dtype="float64")
        for metric, values in dense.items()
    }
    purchase_mask = dense["baskets"] > 0
    day_axis = np.arange(maximum_day + 1, dtype="int16")[None, :]
    first_purchase = np.where(purchase_mask, day_axis, maximum_day + 1).min(axis=1)
    last_purchase, previous_purchase = _purchase_state(purchase_mask)

    frames: list[pd.DataFrame] = []
    for snapshot in snapshots:
        recency = np.where(
            last_purchase[:, snapshot] >= 0,
            snapshot - last_purchase[:, snapshot],
            config.minimum_history_days,
        ).astype("float32")
        payload: dict[str, Any] = {
            "household_key": households,
            "snapshot_day": np.full(len(households), snapshot, dtype="int16"),
            "recency_days": recency,
            "tenure_days": np.maximum(snapshot - first_purchase, 0).astype("float32"),
            "last_interpurchase_gap": np.where(
                previous_purchase[:, snapshot] >= 0,
                last_purchase[:, snapshot] - previous_purchase[:, snapshot],
                config.minimum_history_days,
            ).astype("float32"),
        }

        for window in WINDOWS:
            for metric in ADDITIVE_DAILY_METRICS:
                payload[f"{metric}_{window}d"] = _window(
                    cumulative[metric], snapshot, window
                ).astype("float32")
            baskets = payload[f"baskets_{window}d"]
            spend = payload[f"spend_{window}d"]
            active = payload[f"active_days_{window}d"]
            lines = payload[f"line_items_{window}d"]
            total_value = spend + payload[f"discount_{window}d"]
            payload[f"avg_basket_value_{window}d"] = _safe_ratio(spend, baskets)
            payload[f"lines_per_basket_{window}d"] = _safe_ratio(lines, baskets)
            payload[f"spend_per_active_day_{window}d"] = _safe_ratio(spend, active)
            payload[f"discount_rate_{window}d"] = _safe_ratio(
                payload[f"discount_{window}d"], total_value
            )
            payload[f"coupon_rate_{window}d"] = _safe_ratio(
                payload[f"coupon_discount_{window}d"], total_value
            )
            payload[f"discounted_line_rate_{window}d"] = _safe_ratio(
                payload[f"discounted_lines_{window}d"], lines
            )
            for metric in (
                "private_label_spend",
                "grocery_spend",
                "fresh_spend",
                "drug_gm_spend",
                "fuel_spend",
                "other_spend",
            ):
                payload[f"{metric}_share_{window}d"] = _safe_ratio(
                    payload[f"{metric}_{window}d"], spend
                )
            for metric in ("morning_baskets", "afternoon_baskets", "evening_baskets"):
                payload[f"{metric}_share_{window}d"] = _safe_ratio(
                    payload[f"{metric}_{window}d"], baskets
                )
            for metric in (
                "product_day_diversity",
                "department_day_diversity",
                "store_day_diversity",
            ):
                payload[f"{metric}_per_active_day_{window}d"] = _safe_ratio(
                    payload[f"{metric}_{window}d"], active
                )

        for metric in ("spend", "baskets", "active_days", "line_items"):
            recent = payload[f"{metric}_30d"]
            prior = (
                cumulative[metric][:, max(snapshot - 30, 0)]
                - cumulative[metric][:, max(snapshot - 60, 0)]
            ).astype("float32")
            payload[f"{metric}_trend_30d"] = (
                np.log1p(recent) - np.log1p(prior)
            ).astype("float32")

        lifetime_active = cumulative["active_days"][:, snapshot].astype("float32")
        lifetime_baskets = cumulative["baskets"][:, snapshot].astype("float32")
        lifetime_spend = cumulative["spend"][:, snapshot].astype("float32")
        observed_span = np.maximum(
            last_purchase[:, snapshot] - first_purchase, 0
        ).astype("float32")
        mean_gap = _safe_ratio(observed_span, np.maximum(lifetime_active - 1, 0))
        payload["lifetime_active_days"] = lifetime_active
        payload["lifetime_baskets"] = lifetime_baskets
        payload["lifetime_spend"] = lifetime_spend
        payload["lifetime_avg_basket_value"] = _safe_ratio(
            lifetime_spend, lifetime_baskets
        )
        payload["mean_interpurchase_gap"] = mean_gap
        payload["recency_to_cadence"] = _safe_ratio(recency, mean_gap + 1)
        payload["activity_acceleration_30_vs_90"] = _safe_ratio(
            payload["active_days_30d"] * 3,
            payload["active_days_90d"],
        )

        if include_target:
            future_baskets = (
                cumulative["baskets"][:, snapshot + config.horizon_days]
                - cumulative["baskets"][:, snapshot]
            )
            payload[config.target_column] = (future_baskets > 0).astype("int8")
        known = first_purchase <= snapshot
        frames.append(pd.DataFrame(payload).loc[known].copy())

    table = pd.concat(frames, ignore_index=True)
    table = table.merge(demographics, on="household_key", how="left", validate="many_to_one")
    for column in DEMOGRAPHIC_COLUMNS:
        table[column] = table[column].fillna("Unknown").astype(str)
    table["has_demographics"] = table["has_demographics"].fillna(0).astype("int8")
    profile = {
        "raw_day_min": minimum_day,
        "raw_day_max": maximum_day,
        "households": len(households),
        "daily_rows": len(daily),
        "snapshots": len(snapshots),
        "snapshot_min": min(snapshots),
        "snapshot_max": max(snapshots),
        "modeling_rows": len(table),
        "feature_columns": len(table.columns) - len(ID_COLUMNS) - int(include_target),
        "target_rate": float(table[config.target_column].mean()) if include_target else None,
    }
    return table, profile
