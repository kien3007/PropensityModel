"""
Raw Data Loader: the single entry point through which training, batch scoring and the
feature store read Dunnhumby data.

Two interchangeable sources produce the same `RawDataBundle`:
- "lakehouse": Bronze Parquet tables managed by `LakehouseEngine` (ingested from CSV on demand)
- "csv":       the raw CSV drop in `settings.DATA_RAW_DIR`

Bronze (not Silver) is read on purpose: Silver drops line items with negative
SALES_VALUE, whereas the feature contract nets returns inside a basket and only drops
baskets whose total spend is <= 0. Reading Bronze keeps features identical to the
ones the champion was trained on.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from src.config.settings import settings

if TYPE_CHECKING:
    from src.lakehouse.engine import LakehouseEngine

logger = logging.getLogger(__name__)

TRANSACTION_COLUMNS: List[str] = [
    "household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE", "STORE_ID",
]

# logical name -> (raw CSV file, Bronze Parquet file)
SOURCE_FILES: Dict[str, tuple[str, str]] = {
    "transactions": ("transaction_data.csv", "transactions.parquet"),
    "products": ("product.csv", "products.parquet"),
    "demographics": ("hh_demographic.csv", "demographics.parquet"),
    "campaign_table": ("campaign_table.csv", "campaign_table.parquet"),
    "campaign_desc": ("campaign_desc.csv", "campaign_desc.parquet"),
    "redemptions": ("coupon_redempt.csv", "coupon_redemptions.parquet"),
}


class DataSourceError(FileNotFoundError):
    """Raised when a required raw table cannot be found."""


@dataclass
class RawDataBundle:
    transactions: pd.DataFrame
    products: Optional[pd.DataFrame] = None
    campaigns: Optional[pd.DataFrame] = None  # campaign_table joined with campaign_desc
    redemptions: Optional[pd.DataFrame] = None
    demographics: Optional[pd.DataFrame] = None
    source: str = "memory"
    _fingerprint: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        missing = [c for c in TRANSACTION_COLUMNS if c not in self.transactions.columns]
        if missing:
            raise ValueError(f"transactions is missing required columns: {missing}")
        self.transactions = _normalise_transactions(self.transactions[TRANSACTION_COLUMNS])

    @property
    def households(self) -> List[int]:
        """Scoring universe: every household that appears in the transaction history."""
        return sorted(int(h) for h in self.transactions["household_key"].unique())

    def fingerprint(self) -> str:
        """Content hash of all tables; recorded in model bundles for reproducibility."""
        if self._fingerprint is None:
            digest = hashlib.sha256()
            for name in ("transactions", "products", "campaigns", "redemptions", "demographics"):
                df = getattr(self, name)
                digest.update(name.encode())
                if df is not None:
                    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
            self._fingerprint = digest.hexdigest()[:16]
        return self._fingerprint


def _normalise_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes dtypes so CSV- and Parquet-loaded transactions are indistinguishable.
    SALES_VALUE is rounded to cents: the raw CSV holds float noise (e.g. 5.5511150000000004e-17)
    that pandas and DuckDB parse to different last bits, which would change the data fingerprint.
    """
    tx = tx.assign(SALES_VALUE=tx["SALES_VALUE"].astype("float64").round(2))
    return tx.astype(
        {
            "household_key": "int64",
            "BASKET_ID": "int64",
            "DAY": "int64",
            "PRODUCT_ID": "int64",
            "QUANTITY": "int64",
            "SALES_VALUE": "float64",
            "STORE_ID": "int64",
        }
    ).reset_index(drop=True)


def _join_campaigns(table: Optional[pd.DataFrame], desc: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if table is None or desc is None:
        return None
    return table.merge(desc, on="CAMPAIGN", how="inner")


def _normalise_demographics(demo: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if demo is None:
        return None
    # Keep every attribute as string: the CSV parser and DuckDB type mixed columns
    # (e.g. classification_4 = "1".."5+") differently.
    out = demo.copy()
    for col in out.columns:
        if col != "household_key":
            out[col] = out[col].astype(object).where(out[col].notna(), None)
            out[col] = out[col].map(lambda v: None if v is None else str(v))
    out["household_key"] = out["household_key"].astype("int64")
    return out


class DataLoader:
    """Builds a `RawDataBundle` from CSV or from the Lakehouse Bronze layer."""

    @staticmethod
    def from_csv(raw_dir: Optional[Path | str] = None) -> RawDataBundle:
        raw = Path(raw_dir or settings.DATA_RAW_DIR)
        tx_path = raw / SOURCE_FILES["transactions"][0]
        if not tx_path.exists():
            raise DataSourceError(f"Transactions CSV not found at {tx_path}")

        def read(name: str) -> Optional[pd.DataFrame]:
            path = raw / SOURCE_FILES[name][0]
            return pd.read_csv(path) if path.exists() else None

        logger.info(f"Loading raw data from CSV directory {raw}...")
        return RawDataBundle(
            transactions=pd.read_csv(tx_path, usecols=TRANSACTION_COLUMNS),
            products=read("products"),
            campaigns=_join_campaigns(read("campaign_table"), read("campaign_desc")),
            redemptions=read("redemptions"),
            demographics=_normalise_demographics(read("demographics")),
            source=f"csv:{raw}",
        )

    @staticmethod
    def from_lakehouse(lakehouse: Optional["LakehouseEngine"] = None, ingest: bool = True) -> RawDataBundle:
        from src.lakehouse.engine import LakehouseEngine

        engine = lakehouse or LakehouseEngine()
        if ingest:
            engine.ingest_bronze()
        bronze = engine.bronze_dir
        tx_path = bronze / SOURCE_FILES["transactions"][1]
        if not tx_path.exists():
            raise DataSourceError(f"Bronze transactions not found at {tx_path}; run LakehouseEngine.ingest_bronze()")

        def read(name: str, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
            path = bronze / SOURCE_FILES[name][1]
            return pd.read_parquet(path, columns=columns) if path.exists() else None

        logger.info(f"Loading raw data from Lakehouse Bronze layer {bronze}...")
        return RawDataBundle(
            transactions=read("transactions", TRANSACTION_COLUMNS),
            products=read("products"),
            campaigns=_join_campaigns(read("campaign_table"), read("campaign_desc")),
            redemptions=read("redemptions"),
            demographics=_normalise_demographics(read("demographics")),
            source=f"lakehouse:{bronze}",
        )

    @staticmethod
    def load(source: Optional[str] = None) -> RawDataBundle:
        chosen = (source or settings.DATA_SOURCE).lower()
        if chosen == "lakehouse":
            return DataLoader.from_lakehouse()
        if chosen == "csv":
            return DataLoader.from_csv()
        raise ValueError(f"Unknown DATA_SOURCE '{chosen}' (expected 'lakehouse' or 'csv')")
