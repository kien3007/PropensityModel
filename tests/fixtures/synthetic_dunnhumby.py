"""
Deterministic, small Dunnhumby-shaped dataset for fast tests (no real CSVs needed).

Schemas mirror "The Complete Journey" CSVs exactly (column names and order), so the
same fixture can be written to disk and pushed through the CSV and Lakehouse loaders.
Households have heterogeneous shopping cadences, so purchase propensity is learnable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

DEPARTMENTS = ["PRODUCE", "MEAT", "DAIRY", "GROCERY", "DRUG GM", "KIOSK-GAS"]


def make_raw_tables(n_households: int = 60, n_days: int = 200, seed: int = 7) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    products = pd.DataFrame({
        "PRODUCT_ID": np.arange(1000, 1060),
        "MANUFACTURER": rng.integers(1, 20, 60),
        "DEPARTMENT": [DEPARTMENTS[i % len(DEPARTMENTS)] for i in range(60)],
        "BRAND": rng.choice(["National", "Private"], 60),
        "COMMODITY_DESC": "COMMODITY",
        "SUB_COMMODITY_DESC": "SUB",
        "CURR_SIZE_OF_PRODUCT": "1 LB",
    })

    rows = []
    basket_id = 10_000
    for hh in range(1, n_households + 1):
        mean_gap = rng.choice([2, 4, 7, 14, 30])
        store = int(rng.integers(300, 305))
        day = int(rng.integers(1, mean_gap + 1))
        while day <= n_days:
            basket_id += 1
            for product in rng.choice(products["PRODUCT_ID"], size=int(rng.integers(1, 6)), replace=False):
                rows.append((hh, basket_id, day, int(product), int(rng.integers(1, 4)),
                             round(float(rng.uniform(0.5, 12.0)), 2), store, 0.0, 1200, (day + 1) // 7 + 1, 0.0, 0.0))
            day += max(1, int(rng.poisson(mean_gap)))
    transactions = pd.DataFrame(rows, columns=[
        "household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE", "STORE_ID",
        "RETAIL_DISC", "TRANS_TIME", "WEEK_NO", "COUPON_DISC", "COUPON_MATCH_DISC",
    ])
    # One fully-returned basket (total spend <= 0) that must be ignored by features and labels.
    transactions.loc[len(transactions)] = (1, 1, n_days // 2, 1000, 1, -3.0, 300, 0.0, 1200, 1, 0.0, 0.0)

    demo_hh = np.arange(1, n_households + 1, 2)
    demographics = pd.DataFrame({
        "classification_1": rng.choice(["Age Group1", "Age Group4", "Age Group6"], len(demo_hh)),
        "classification_2": rng.choice(["X", "Y", "Z"], len(demo_hh)),
        "classification_3": rng.choice(["Level1", "Level4"], len(demo_hh)),
        "HOMEOWNER_DESC": rng.choice(["Homeowner", "Renter", "Unknown"], len(demo_hh)),
        "classification_5": rng.choice(["Group1", "Group5"], len(demo_hh)),
        "classification_4": rng.choice(["1", "2", "5+"], len(demo_hh)),
        "KID_CATEGORY_DESC": rng.choice(["1", "None/Unknown"], len(demo_hh)),
        "household_key": demo_hh,
    })

    campaign_desc = pd.DataFrame({
        "DESCRIPTION": ["TypeA", "TypeB"],
        "CAMPAIGN": [1, 2],
        "START_DAY": [30, n_days // 2],
        "END_DAY": [90, n_days],
    })
    campaign_table = pd.DataFrame({
        "DESCRIPTION": ["TypeA"] * 10 + ["TypeB"] * 10,
        "household_key": list(range(1, 11)) + list(range(11, 21)),
        "CAMPAIGN": [1] * 10 + [2] * 10,
    })
    redemptions = pd.DataFrame({
        "household_key": [1, 2, 3, 3, 11],
        "DAY": [40, 50, 60, 150, n_days - 5],
        "COUPON_UPC": [111, 222, 333, 444, 555],
        "CAMPAIGN": [1, 1, 1, 2, 2],
    })
    return {
        "transactions": transactions,
        "products": products,
        "demographics": demographics,
        "campaign_table": campaign_table,
        "campaign_desc": campaign_desc,
        "redemptions": redemptions,
    }


CSV_NAMES = {
    "transactions": "transaction_data.csv",
    "products": "product.csv",
    "demographics": "hh_demographic.csv",
    "campaign_table": "campaign_table.csv",
    "campaign_desc": "campaign_desc.csv",
    "redemptions": "coupon_redempt.csv",
}


def write_raw_csvs(tables: Dict[str, pd.DataFrame], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(directory / CSV_NAMES[name], index=False)
    return directory
