"""Comprehensive Phase-2 data audit for Dunnhumby Complete Journey.

The audit is intentionally separate from model training.  It produces stable,
reviewable evidence for target validity, data quality, joins, temporal coverage,
household behavior, product mix, campaigns, coupons, and causal media data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRANSACTION_DTYPES = {
    "household_key": "int32",
    "BASKET_ID": "int64",
    "DAY": "int16",
    "PRODUCT_ID": "int64",
    "QUANTITY": "int32",
    "SALES_VALUE": "float32",
    "STORE_ID": "int32",
    "RETAIL_DISC": "float32",
    "TRANS_TIME": "int16",
    "WEEK_NO": "int16",
    "COUPON_DISC": "float32",
    "COUPON_MATCH_DISC": "float32",
}


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_native(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def describe_series(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    quantiles = numeric.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
    return {
        "count": int(numeric.size),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()),
        "min": float(quantiles.loc[0.0]),
        "p01": float(quantiles.loc[0.01]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1.0]),
    }


def load_transactions(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / "transaction_data.csv",
        dtype=TRANSACTION_DTYPES,
    )


def inventory_small_files(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    loaded: dict[str, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.csv")):
        if path.name in {"transaction_data.csv", "causal_data.csv"}:
            continue
        frame = pd.read_csv(path)
        loaded[path.stem] = frame
        records.append(
            {
                "file": path.name,
                "size_mb": path.stat().st_size / (1024**2),
                "rows": len(frame),
                "columns": len(frame.columns),
                "duplicate_rows": int(frame.duplicated().sum()),
                "missing_cells": int(frame.isna().sum().sum()),
                "column_names": "|".join(frame.columns),
            }
        )
    return pd.DataFrame(records), loaded


def audit_causal_data(path: Path, chunksize: int = 1_000_000) -> dict[str, Any]:
    rows = 0
    missing = Counter()
    display = Counter()
    mailer = Counter()
    products: set[int] = set()
    stores: set[int] = set()
    weeks: set[int] = set()
    minimum_week: int | None = None
    maximum_week: int | None = None
    causal_dtypes = {
        "PRODUCT_ID": "int64",
        "STORE_ID": "int32",
        "WEEK_NO": "int16",
        "display": "string",
        "mailer": "string",
    }
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=causal_dtypes):
        rows += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        display.update(chunk["display"].fillna("<MISSING>").astype(str).value_counts().to_dict())
        mailer.update(chunk["mailer"].fillna("<MISSING>").astype(str).value_counts().to_dict())
        products.update(chunk["PRODUCT_ID"].dropna().astype(int).unique().tolist())
        stores.update(chunk["STORE_ID"].dropna().astype(int).unique().tolist())
        chunk_weeks = chunk["WEEK_NO"].dropna().astype(int)
        weeks.update(chunk_weeks.unique().tolist())
        if not chunk_weeks.empty:
            chunk_min = int(chunk_weeks.min())
            chunk_max = int(chunk_weeks.max())
            minimum_week = chunk_min if minimum_week is None else min(minimum_week, chunk_min)
            maximum_week = chunk_max if maximum_week is None else max(maximum_week, chunk_max)
    return {
        "rows": rows,
        "columns": 5,
        "missing_by_column": dict(missing),
        "unique_products": len(products),
        "unique_stores": len(stores),
        "unique_weeks": len(weeks),
        "week_min": minimum_week,
        "week_max": maximum_week,
        "display_distribution": dict(display),
        "mailer_distribution": dict(mailer),
    }


def transaction_quality(transactions: pd.DataFrame) -> dict[str, Any]:
    trans_time = transactions["TRANS_TIME"].astype("int32")
    invalid_time = (trans_time < 0) | (trans_time > 2359) | ((trans_time % 100) >= 60)
    basket_integrity = transactions.groupby("BASKET_ID", observed=True).agg(
        households=("household_key", "nunique"),
        days=("DAY", "nunique"),
        stores=("STORE_ID", "nunique"),
        times=("TRANS_TIME", "nunique"),
    )
    return {
        "rows": len(transactions),
        "columns": len(transactions.columns),
        "missing_by_column": transactions.isna().sum().to_dict(),
        "exact_duplicate_rows": int(transactions.duplicated().sum()),
        "households": int(transactions["household_key"].nunique()),
        "baskets": int(transactions["BASKET_ID"].nunique()),
        "products": int(transactions["PRODUCT_ID"].nunique()),
        "stores": int(transactions["STORE_ID"].nunique()),
        "day_min": int(transactions["DAY"].min()),
        "day_max": int(transactions["DAY"].max()),
        "week_min": int(transactions["WEEK_NO"].min()),
        "week_max": int(transactions["WEEK_NO"].max()),
        "negative_quantity_rows": int((transactions["QUANTITY"] < 0).sum()),
        "zero_quantity_rows": int((transactions["QUANTITY"] == 0).sum()),
        "negative_sales_rows": int((transactions["SALES_VALUE"] < 0).sum()),
        "zero_sales_rows": int((transactions["SALES_VALUE"] == 0).sum()),
        "positive_discount_rows": int((transactions["RETAIL_DISC"] > 0).sum()),
        "invalid_trans_time_rows": int(invalid_time.sum()),
        "multi_household_baskets": int((basket_integrity["households"] > 1).sum()),
        "multi_day_baskets": int((basket_integrity["days"] > 1).sum()),
        "multi_store_baskets": int((basket_integrity["stores"] > 1).sum()),
        "multi_time_baskets": int((basket_integrity["times"] > 1).sum()),
        "quantity_distribution": describe_series(transactions["QUANTITY"]),
        "sales_value_distribution": describe_series(transactions["SALES_VALUE"]),
    }


def build_behavior_tables(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = transactions.copy()
    frame["discount"] = -(
        frame["RETAIL_DISC"] + frame["COUPON_DISC"] + frame["COUPON_MATCH_DISC"]
    ).clip(upper=0)
    frame["has_discount"] = (frame["discount"] > 0).astype("int8")
    basket = (
        frame.groupby(
            ["household_key", "BASKET_ID", "DAY", "WEEK_NO", "STORE_ID", "TRANS_TIME"],
            observed=True,
            sort=False,
        )
        .agg(
            sales=("SALES_VALUE", "sum"),
            units=("QUANTITY", "sum"),
            discount=("discount", "sum"),
            product_lines=("PRODUCT_ID", "size"),
            unique_products=("PRODUCT_ID", "nunique"),
            discounted_lines=("has_discount", "sum"),
        )
        .reset_index()
    )
    daily = (
        basket.groupby(["household_key", "DAY"], observed=True, sort=True)
        .agg(
            spend=("sales", "sum"),
            units=("units", "sum"),
            discount=("discount", "sum"),
            baskets=("BASKET_ID", "nunique"),
            stores=("STORE_ID", "nunique"),
        )
        .reset_index()
    )
    daily["active_day"] = 1

    purchase_days = daily[["household_key", "DAY"]].sort_values(["household_key", "DAY"])
    purchase_days["gap_days"] = purchase_days.groupby("household_key")["DAY"].diff()
    gap_summary = purchase_days.groupby("household_key", observed=True)["gap_days"].agg(
        mean_gap_days="mean",
        median_gap_days="median",
        std_gap_days="std",
        max_gap_days="max",
    )
    household = (
        frame.groupby("household_key", observed=True)
        .agg(
            first_day=("DAY", "min"),
            last_day=("DAY", "max"),
            line_items=("PRODUCT_ID", "size"),
            unique_products=("PRODUCT_ID", "nunique"),
            stores=("STORE_ID", "nunique"),
            spend=("SALES_VALUE", "sum"),
            units=("QUANTITY", "sum"),
            discount=("discount", "sum"),
        )
        .join(
            basket.groupby("household_key", observed=True).agg(
                baskets=("BASKET_ID", "nunique"),
                avg_basket_value=("sales", "mean"),
                median_basket_value=("sales", "median"),
            )
        )
        .join(
            daily.groupby("household_key", observed=True).agg(active_days=("DAY", "nunique"))
        )
        .join(gap_summary)
        .reset_index()
    )
    household["observed_tenure_days"] = household["last_day"] - household["first_day"] + 1
    household["discount_to_sales"] = household["discount"] / (
        household["spend"] + household["discount"]
    ).replace(0, np.nan)
    return basket, daily, household


def target_horizon_analysis(
    daily: pd.DataFrame,
    horizons: tuple[int, ...] = (7, 14, 30),
    history_days: int = 180,
    snapshot_step: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    households = np.sort(daily["household_key"].unique())
    maximum_day = int(daily["DAY"].max())
    basket_matrix = (
        daily.pivot(index="household_key", columns="DAY", values="baskets")
        .reindex(index=households, columns=range(maximum_day + 1), fill_value=0)
        .fillna(0)
        .to_numpy(dtype="float32")
    )
    cumulative = np.cumsum(basket_matrix, axis=1, dtype="float64")
    purchase_mask = basket_matrix > 0
    day_axis = np.arange(maximum_day + 1, dtype="int16")[None, :]
    first_purchase = np.where(purchase_mask, day_axis, maximum_day + 1).min(axis=1)
    last_purchase = np.maximum.accumulate(np.where(purchase_mask, day_axis, -1), axis=1)

    snapshot_records: list[dict[str, Any]] = []
    for horizon in horizons:
        snapshots = range(history_days, maximum_day - horizon + 1, snapshot_step)
        for snapshot in snapshots:
            eligible = first_purchase <= snapshot
            target = (
                cumulative[:, snapshot + horizon] - cumulative[:, snapshot]
            )[eligible] > 0
            recency = snapshot - last_purchase[:, snapshot][eligible]
            snapshot_records.append(
                {
                    "horizon_days": horizon,
                    "snapshot_day": snapshot,
                    "eligible_households": int(eligible.sum()),
                    "positive_households": int(target.sum()),
                    "target_rate": float(target.mean()),
                    "recency_median": float(np.median(recency)),
                }
            )
    snapshot_table = pd.DataFrame(snapshot_records)
    summary = (
        snapshot_table.groupby("horizon_days", observed=True)
        .agg(
            snapshots=("snapshot_day", "nunique"),
            modeling_rows=("eligible_households", "sum"),
            mean_target_rate=("target_rate", "mean"),
            min_target_rate=("target_rate", "min"),
            max_target_rate=("target_rate", "max"),
            std_target_rate=("target_rate", "std"),
            first_snapshot=("snapshot_day", "min"),
            last_snapshot=("snapshot_day", "max"),
        )
        .reset_index()
    )
    return snapshot_table, summary


def audit_relations(
    transactions: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    product = tables["product"]
    demographics = tables["hh_demographic"]
    campaign_table = tables["campaign_table"]
    campaign_desc = tables["campaign_desc"]
    coupon = tables["coupon"]
    redemption = tables["coupon_redempt"]

    product_ids = set(product["PRODUCT_ID"])
    household_ids = set(transactions["household_key"])
    campaign_ids = set(campaign_desc["CAMPAIGN"])
    coupon_ids = set(coupon["COUPON_UPC"])
    transaction_products = transactions["PRODUCT_ID"].isin(product_ids)

    redemption_campaign = redemption.merge(
        campaign_desc[["CAMPAIGN", "START_DAY", "END_DAY"]],
        on="CAMPAIGN",
        how="left",
        validate="many_to_one",
    )
    redemption_in_window = redemption_campaign["DAY"].between(
        redemption_campaign["START_DAY"], redemption_campaign["END_DAY"]
    )

    enriched = transactions[[
        "household_key",
        "BASKET_ID",
        "PRODUCT_ID",
        "QUANTITY",
        "SALES_VALUE",
        "RETAIL_DISC",
        "COUPON_DISC",
        "COUPON_MATCH_DISC",
    ]].merge(
        product[["PRODUCT_ID", "DEPARTMENT", "BRAND", "COMMODITY_DESC"]],
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )
    enriched["discount"] = -(
        enriched["RETAIL_DISC"]
        + enriched["COUPON_DISC"]
        + enriched["COUPON_MATCH_DISC"]
    ).clip(upper=0)
    department = (
        enriched.groupby("DEPARTMENT", dropna=False, observed=True)
        .agg(
            line_items=("PRODUCT_ID", "size"),
            products=("PRODUCT_ID", "nunique"),
            households=("household_key", "nunique"),
            baskets=("BASKET_ID", "nunique"),
            sales=("SALES_VALUE", "sum"),
            units=("QUANTITY", "sum"),
            discount=("discount", "sum"),
        )
        .reset_index()
        .sort_values("sales", ascending=False)
    )
    department["sales_share"] = department["sales"] / department["sales"].sum()

    brand = (
        enriched.groupby("BRAND", dropna=False, observed=True)
        .agg(
            line_items=("PRODUCT_ID", "size"),
            products=("PRODUCT_ID", "nunique"),
            households=("household_key", "nunique"),
            sales=("SALES_VALUE", "sum"),
            discount=("discount", "sum"),
        )
        .reset_index()
        .sort_values("sales", ascending=False)
    )
    brand["sales_share"] = brand["sales"] / brand["sales"].sum()

    relations = {
        "transaction_product_match_rate": float(transaction_products.mean()),
        "transaction_households_with_demographics": int(
            transactions.loc[
                transactions["household_key"].isin(set(demographics["household_key"])),
                "household_key",
            ].nunique()
        ),
        "demographic_household_coverage_rate": float(
            len(set(demographics["household_key"]) & household_ids) / len(household_ids)
        ),
        "campaign_household_coverage_rate": float(
            len(set(campaign_table["household_key"]) & household_ids) / len(household_ids)
        ),
        "campaign_table_unknown_campaign_rows": int(
            (~campaign_table["CAMPAIGN"].isin(campaign_ids)).sum()
        ),
        "coupon_unknown_product_rows": int((~coupon["PRODUCT_ID"].isin(product_ids)).sum()),
        "redemption_unknown_household_rows": int(
            (~redemption["household_key"].isin(household_ids)).sum()
        ),
        "redemption_unknown_coupon_rows": int(
            (~redemption["COUPON_UPC"].isin(coupon_ids)).sum()
        ),
        "redemption_with_known_campaign_rate": float(
            redemption_campaign["START_DAY"].notna().mean()
        ),
        "redemption_within_campaign_window_rate": float(redemption_in_window.mean()),
    }
    return relations, department, brand


def build_weekly_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    frame = transactions.copy()
    frame["discount"] = -(
        frame["RETAIL_DISC"] + frame["COUPON_DISC"] + frame["COUPON_MATCH_DISC"]
    ).clip(upper=0)
    return (
        frame.groupby("WEEK_NO", observed=True)
        .agg(
            line_items=("PRODUCT_ID", "size"),
            households=("household_key", "nunique"),
            baskets=("BASKET_ID", "nunique"),
            products=("PRODUCT_ID", "nunique"),
            sales=("SALES_VALUE", "sum"),
            units=("QUANTITY", "sum"),
            discount=("discount", "sum"),
        )
        .reset_index()
        .sort_values("WEEK_NO")
    )


def demographic_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for column in frame.columns:
        if column == "household_key":
            continue
        counts = frame[column].fillna("<MISSING>").astype(str).value_counts(dropna=False)
        for value, count in counts.items():
            records.append(
                {
                    "column": column,
                    "value": value,
                    "households": int(count),
                    "share": float(count / len(frame)),
                }
            )
    return pd.DataFrame(records)


def run_audit(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory, tables = inventory_small_files(raw_dir)
    transactions = load_transactions(raw_dir)
    transaction_profile = transaction_quality(transactions)
    basket, daily, household = build_behavior_tables(transactions)
    relations, department, brand = audit_relations(transactions, tables)
    weekly = build_weekly_summary(transactions)
    target_snapshots, target_summary = target_horizon_analysis(daily)
    demographics = demographic_summary(tables["hh_demographic"])
    causal = audit_causal_data(raw_dir / "causal_data.csv")

    inventory = pd.concat(
        [
            inventory,
            pd.DataFrame(
                [
                    {
                        "file": "transaction_data.csv",
                        "size_mb": (raw_dir / "transaction_data.csv").stat().st_size
                        / (1024**2),
                        "rows": len(transactions),
                        "columns": len(transactions.columns),
                        "duplicate_rows": transaction_profile["exact_duplicate_rows"],
                        "missing_cells": sum(transaction_profile["missing_by_column"].values()),
                        "column_names": "|".join(transactions.columns),
                    },
                    {
                        "file": "causal_data.csv",
                        "size_mb": (raw_dir / "causal_data.csv").stat().st_size / (1024**2),
                        "rows": causal["rows"],
                        "columns": causal["columns"],
                        "duplicate_rows": np.nan,
                        "missing_cells": sum(causal["missing_by_column"].values()),
                        "column_names": "PRODUCT_ID|STORE_ID|WEEK_NO|display|mailer",
                    },
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("file")

    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)
    basket.to_parquet(output_dir / "basket_summary.parquet", index=False)
    household.to_csv(output_dir / "household_summary.csv", index=False)
    weekly.to_csv(output_dir / "weekly_summary.csv", index=False)
    department.to_csv(output_dir / "department_summary.csv", index=False)
    brand.to_csv(output_dir / "brand_summary.csv", index=False)
    demographics.to_csv(output_dir / "demographic_summary.csv", index=False)
    target_snapshots.to_csv(output_dir / "target_snapshot_analysis.csv", index=False)
    target_summary.to_csv(output_dir / "target_horizon_summary.csv", index=False)

    issues = [
        {
            "severity": "high",
            "area": "business_semantics",
            "issue": "Propensity is not incremental response to a campaign.",
            "action": "Use randomized treatment/control data before optimizing coupon uplift.",
        },
        {
            "severity": "medium",
            "area": "demographics",
            "issue": (
                f"Demographics cover only {relations['demographic_household_coverage_rate']:.1%} "
                "of transaction households."
            ),
            "action": (
                "Retain Unknown explicitly and report segment metrics only with "
                "adequate support."
            ),
        },
        {
            "severity": "medium",
            "area": "time",
            "issue": "DAY is an integer index without a calendar-date mapping.",
            "action": (
                "Avoid claims about real-world seasonality; validate calendar mapping "
                "if available."
            ),
        },
        {
            "severity": "high",
            "area": "quantity_semantics",
            "issue": (
                "QUANTITY uses a scaled convention for fuel and miscellaneous sales; "
                "it is not a comparable count of physical items."
            ),
            "action": "Exclude raw QUANTITY from propensity V2; use baskets, lines and diversity.",
        },
        {
            "severity": "medium",
            "area": "causal_data",
            "issue": "Causal media data is product-store-week level and very large.",
            "action": (
                "Join only point-in-time media exposure aggregates needed by a "
                "defined use case."
            ),
        },
    ]
    pd.DataFrame(issues).to_csv(output_dir / "data_quality_issues.csv", index=False)

    report = {
        "transaction_quality": transaction_profile,
        "relational_integrity_and_coverage": relations,
        "causal_data": causal,
        "household_behavior": {
            "households": len(household),
            "active_days": describe_series(household["active_days"]),
            "baskets": describe_series(household["baskets"]),
            "spend": describe_series(household["spend"]),
            "mean_gap_days": describe_series(household["mean_gap_days"]),
            "avg_basket_value": describe_series(household["avg_basket_value"]),
        },
        "target_horizons": target_summary.to_dict(orient="records"),
    }
    write_json(output_dir / "data_audit.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("sample_data/dunnhumby"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/propensity_v2/data_understanding"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(args.raw_dir, args.output_dir)
    print(json.dumps(_native(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
