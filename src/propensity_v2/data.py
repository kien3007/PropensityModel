"""Raw-data loading and household-day feature source construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ADDITIVE_DAILY_METRICS, DEMOGRAPHIC_COLUMNS


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

FRESH_DEPARTMENTS = {
    "PRODUCE",
    "MEAT",
    "MEAT-PCKGD",
    "DELI",
    "PASTRY",
    "SEAFOOD",
    "SEAFOOD-PCKGD",
    "SALAD BAR",
}


def load_transactions(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "transaction_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing transaction data: {path}")
    return pd.read_csv(path, dtype=TRANSACTION_DTYPES)


def load_demographics(raw_dir: Path, households: np.ndarray) -> pd.DataFrame:
    path = raw_dir / "hh_demographic.csv"
    base = pd.DataFrame({"household_key": households.astype("int32")})
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
    result["has_demographics"] = (
        result[DEMOGRAPHIC_COLUMNS[0]] != "Unknown"
    ).astype("int8")
    return result


def enrich_transactions(transactions: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    product = pd.read_csv(
        raw_dir / "product.csv",
        usecols=["PRODUCT_ID", "DEPARTMENT", "BRAND", "COMMODITY_DESC"],
        dtype={"PRODUCT_ID": "int64"},
    )
    frame = transactions.merge(product, on="PRODUCT_ID", how="left", validate="many_to_one")
    if frame["DEPARTMENT"].isna().any():
        raise ValueError("Some transaction products do not match product.csv")

    frame["discount"] = -(
        frame["RETAIL_DISC"] + frame["COUPON_DISC"] + frame["COUPON_MATCH_DISC"]
    ).clip(upper=0)
    frame["coupon_discount"] = -(
        frame["COUPON_DISC"] + frame["COUPON_MATCH_DISC"]
    ).clip(upper=0)
    frame["discounted_lines"] = (frame["discount"] > 0).astype("int8")
    frame["private_label_spend"] = np.where(
        frame["BRAND"].eq("Private"), frame["SALES_VALUE"], 0
    ).astype("float32")

    department = frame["DEPARTMENT"].fillna("UNKNOWN")
    frame["grocery_spend"] = np.where(
        department.eq("GROCERY"), frame["SALES_VALUE"], 0
    ).astype("float32")
    frame["fresh_spend"] = np.where(
        department.isin(FRESH_DEPARTMENTS), frame["SALES_VALUE"], 0
    ).astype("float32")
    frame["drug_gm_spend"] = np.where(
        department.eq("DRUG GM"), frame["SALES_VALUE"], 0
    ).astype("float32")
    frame["fuel_spend"] = np.where(
        department.eq("KIOSK-GAS"), frame["SALES_VALUE"], 0
    ).astype("float32")
    known_group = department.eq("GROCERY") | department.isin(FRESH_DEPARTMENTS) | department.isin(
        ["DRUG GM", "KIOSK-GAS"]
    )
    frame["other_spend"] = np.where(
        ~known_group, frame["SALES_VALUE"], 0
    ).astype("float32")
    return frame


def build_daily_features(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe additive metrics at household-day grain.

    QUANTITY is deliberately excluded because fuel and miscellaneous-sale rows use
    a scaled quantity convention (up to 89,638), which is not comparable with item
    counts in grocery departments.
    """

    aggregations = {
        "spend": ("SALES_VALUE", "sum"),
        "baskets": ("BASKET_ID", "nunique"),
        "line_items": ("PRODUCT_ID", "size"),
        "discount": ("discount", "sum"),
        "coupon_discount": ("coupon_discount", "sum"),
        "discounted_lines": ("discounted_lines", "sum"),
        "private_label_spend": ("private_label_spend", "sum"),
        "grocery_spend": ("grocery_spend", "sum"),
        "fresh_spend": ("fresh_spend", "sum"),
        "drug_gm_spend": ("drug_gm_spend", "sum"),
        "fuel_spend": ("fuel_spend", "sum"),
        "other_spend": ("other_spend", "sum"),
        "product_day_diversity": ("PRODUCT_ID", "nunique"),
        "department_day_diversity": ("DEPARTMENT", "nunique"),
        "store_day_diversity": ("STORE_ID", "nunique"),
    }
    daily = (
        enriched.groupby(["household_key", "DAY"], observed=True, sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    daily["active_days"] = 1.0

    basket = (
        enriched.groupby(
            ["household_key", "BASKET_ID", "DAY"], observed=True, sort=False
        )
        .agg(
            basket_spend=("SALES_VALUE", "sum"),
            basket_lines=("PRODUCT_ID", "size"),
            trans_time=("TRANS_TIME", "first"),
            store_id=("STORE_ID", "first"),
        )
        .reset_index()
    )
    basket["daypart"] = pd.cut(
        basket["trans_time"],
        bins=[-1, 1159, 1759, 2359],
        labels=["morning", "afternoon", "evening"],
    )
    daypart = (
        basket.pivot_table(
            index=["household_key", "DAY"],
            columns="daypart",
            values="BASKET_ID",
            aggfunc="nunique",
            fill_value=0,
            observed=True,
        )
        .rename(columns=lambda value: f"{value}_baskets")
        .reset_index()
    )
    daily = daily.merge(daypart, on=["household_key", "DAY"], how="left")
    for column in ("morning_baskets", "afternoon_baskets", "evening_baskets"):
        if column not in daily:
            daily[column] = 0
        daily[column] = daily[column].fillna(0)
    for metric in ADDITIVE_DAILY_METRICS:
        daily[metric] = daily[metric].astype("float32")
    return daily, basket
