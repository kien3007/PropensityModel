"""
Point-in-Time Feature Extraction.

All features for a cutoff t0 are computed strictly from data with DAY <= t0.
The 28-feature multi-domain set (src/features/contract.py) is computed by the pure
function `compute_multidomain_features`, which takes DataFrames and performs no I/O.
`MultiDomainFeatureBuilder` wraps it for repeated cutoffs over one loaded dataset and
is the single code path used by training, batch scoring and the feature store.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, List, Optional

import pandas as pd

from src.features.contract import (
    CATEGORICAL_FEATURES,
    FEATURE_SET_VERSION,
    WINDOW_INCLUDES_START_DAY,
    features_for,
    normalize_feature_dtypes,
    validate_feature_frame,
)

if TYPE_CHECKING:
    from src.data.loader import RawDataBundle

logger = logging.getLogger(__name__)

PERISHABLE_DEPARTMENTS: List[str] = [
    "PRODUCE", "MEAT", "MEAT-PCKGD", "DELI", "PASTRY", "DAIRY"
]


class PointInTimeFeatureExtractor:
    """Trip-level basket table that every feature (and label) is computed from."""

    @staticmethod
    def build_trip_level_baskets(
        transactions: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        perishable_departments: List[str] = PERISHABLE_DEPARTMENTS,
    ) -> pd.DataFrame:
        """
        Aggregates raw item transactions into trip-level baskets with perishable indicators.
        Baskets with non-positive total spend (voids / pure returns) are dropped.
        """
        perish_ids = set()
        if products is not None and "DEPARTMENT" in products.columns and "PRODUCT_ID" in products.columns:
            perish_ids = set(products[products["DEPARTMENT"].isin(perishable_departments)]["PRODUCT_ID"])

        tx = transactions.copy()
        tx["is_perishable"] = tx["PRODUCT_ID"].isin(perish_ids) if perish_ids and "PRODUCT_ID" in tx.columns else False

        baskets = tx.groupby(["household_key", "BASKET_ID", "DAY", "STORE_ID"], as_index=False).agg(
            basket_spend=("SALES_VALUE", "sum"),
            basket_items=("QUANTITY", "sum") if "QUANTITY" in tx.columns else ("SALES_VALUE", "size"),
            distinct_items=("PRODUCT_ID", "nunique") if "PRODUCT_ID" in tx.columns else ("SALES_VALUE", "size"),
            has_perishable=("is_perishable", "any"),
        )
        baskets = baskets[baskets["basket_spend"] > 0].copy()
        baskets.sort_values(["household_key", "DAY"], inplace=True)
        baskets["prev_day"] = baskets.groupby("household_key")["DAY"].shift(1)
        baskets["inter_trip_days"] = baskets["DAY"] - baskets["prev_day"]
        return baskets


def compute_multidomain_features(
    baskets: pd.DataFrame,
    cutoff: int,
    households: Iterable[int],
    campaigns: Optional[pd.DataFrame] = None,
    redemptions: Optional[pd.DataFrame] = None,
    demographics: Optional[pd.DataFrame] = None,
    lookback_days: int = 180,
    feature_set_version: str = FEATURE_SET_VERSION,
) -> pd.DataFrame:
    """
    Computes the multi-domain features of `feature_set_version` (Recency/Timing, Frequency/Momentum, Monetary,
    Marketing, Store Loyalty, Demographics) for every household at `cutoff`.

    Pure function: no I/O. Strictly point-in-time: only rows with DAY <= cutoff are used.
    `baskets` comes from `PointInTimeFeatureExtractor.build_trip_level_baskets`;
    `campaigns` is campaign_table joined with campaign_desc (START_DAY / END_DAY).
    """
    model_features = features_for(feature_set_version)
    inclusive = WINDOW_INCLUDES_START_DAY[feature_set_version]

    def in_window(day: pd.Series, n: int) -> pd.Series:
        """Rows of the last n days up to the cutoff (v1 kept the day cutoff - n as well)."""
        return (day >= cutoff - n) if inclusive else (day > cutoff - n)

    hist = baskets[(baskets["DAY"] <= cutoff) & in_window(baskets["DAY"], lookback_days)].copy()
    universe = list(households)
    feats = pd.DataFrame({"household_key": universe})
    feats["cutoff_day"] = cutoff

    # A. Recency & Timing
    latest_trip = hist.groupby("household_key")["DAY"].max().rename("last_trip_day").reset_index()
    feats = feats.merge(latest_trip, on="household_key", how="left")
    feats["days_since_last_basket"] = (cutoff - feats["last_trip_day"]).fillna(lookback_days + 1)

    perish_hist = hist[hist["has_perishable"].astype(bool)]
    if not perish_hist.empty:
        latest_perish = perish_hist.groupby("household_key")["DAY"].max().rename("last_perish_day").reset_index()
        feats = feats.merge(latest_perish, on="household_key", how="left")
        feats["days_since_last_perishable"] = (cutoff - feats["last_perish_day"]).fillna(lookback_days + 1)
    else:
        feats["days_since_last_perishable"] = lookback_days + 1

    inter_stats = hist.groupby("household_key")["inter_trip_days"].agg(["median", "std"]).reset_index()
    inter_stats.columns = ["household_key", "median_inter_trip", "std_inter_trip"]
    feats = feats.merge(inter_stats, on="household_key", how="left")
    feats["median_inter_trip"] = feats["median_inter_trip"].fillna(lookback_days)
    feats["std_inter_trip"] = feats["std_inter_trip"].fillna(0)

    feats["urgency_ratio"] = feats["days_since_last_basket"] / (feats["median_inter_trip"] + 1e-5)

    # B. Frequency & Momentum
    for window in [14, 30, 60, 90, 180]:
        sub = hist[in_window(hist["DAY"], window)]
        counts = sub.groupby("household_key")["BASKET_ID"].nunique().reset_index()
        counts.columns = ["household_key", f"baskets_count_{window}d"]
        feats = feats.merge(counts, on="household_key", how="left")
        feats[f"baskets_count_{window}d"] = feats[f"baskets_count_{window}d"].fillna(0)

    feats["frequency_momentum_30_vs_90"] = (
        (feats["baskets_count_30d"] / 30.0) / ((feats["baskets_count_90d"] / 90.0) + 1e-5)
    )

    # C. Monetary & Basket Architecture
    for window in [30, 90, 180]:
        sub = hist[in_window(hist["DAY"], window)]
        spend = sub.groupby("household_key")["basket_spend"].sum().reset_index()
        spend.columns = ["household_key", f"spend_{window}d"]
        feats = feats.merge(spend, on="household_key", how="left")
        feats[f"spend_{window}d"] = feats[f"spend_{window}d"].fillna(0)

    feats["aov_90d"] = feats["spend_90d"] / (feats["baskets_count_90d"] + 1e-5)

    sub90 = hist[in_window(hist["DAY"], 90)]
    if not sub90.empty:
        avg_items = sub90.groupby("household_key")["basket_items"].mean().reset_index()
        avg_items.columns = ["household_key", "avg_items_per_basket_90d"]
        feats = feats.merge(avg_items, on="household_key", how="left")
        feats["avg_items_per_basket_90d"] = feats["avg_items_per_basket_90d"].fillna(0)

        perish_ratio = sub90.groupby("household_key")["has_perishable"].mean().reset_index()
        perish_ratio.columns = ["household_key", "perishable_basket_ratio_90d"]
        feats = feats.merge(perish_ratio, on="household_key", how="left")
        feats["perishable_basket_ratio_90d"] = feats["perishable_basket_ratio_90d"].fillna(0)
    else:
        feats["avg_items_per_basket_90d"] = 0
        feats["perishable_basket_ratio_90d"] = 0

    # D. Marketing & Coupon
    if campaigns is not None and not campaigns.empty and "START_DAY" in campaigns.columns:
        active_camps = campaigns[(campaigns["START_DAY"] <= cutoff) & (campaigns["END_DAY"] >= cutoff)]
        active_hh = set(active_camps["household_key"].unique())
        feats["is_targeted_by_active_campaign"] = feats["household_key"].isin(active_hh).astype(int)
    else:
        feats["is_targeted_by_active_campaign"] = 0

    if redemptions is not None and not redemptions.empty and "DAY" in redemptions.columns:
        hist_red = redemptions[(redemptions["DAY"] <= cutoff) & in_window(redemptions["DAY"], lookback_days)]
        red_counts = hist_red.groupby("household_key")["DAY"].count().reset_index()
        red_counts.columns = ["household_key", "coupon_redemption_count_180d"]
        feats = feats.merge(red_counts, on="household_key", how="left")
        feats["coupon_redemption_count_180d"] = feats["coupon_redemption_count_180d"].fillna(0)
    else:
        feats["coupon_redemption_count_180d"] = 0

    # E. Store Loyalty
    if not hist.empty and "STORE_ID" in hist.columns:
        store_agg = hist.groupby(["household_key", "STORE_ID"])["basket_spend"].sum().reset_index()
        primary_store = (
            store_agg.sort_values(["household_key", "basket_spend"], ascending=[True, False])
            .groupby("household_key")
            .first()
            .reset_index()
            .rename(columns={"STORE_ID": "primary_store_id", "basket_spend": "primary_store_spend"})
        )
        feats = feats.merge(
            primary_store[["household_key", "primary_store_id", "primary_store_spend"]], on="household_key", how="left"
        )
        feats["store_loyalty_index"] = (
            (feats["primary_store_spend"] / (feats["spend_180d"] + 1e-5)).fillna(0).clip(0, 1)
        )
        feats["primary_store_id"] = feats["primary_store_id"].fillna(0).astype(int)
    else:
        feats["primary_store_id"] = 0
        feats["store_loyalty_index"] = 0

    # F. Demographics (static attributes; missing households -> UNKNOWN)
    if demographics is not None and not demographics.empty:
        avail_cols = [c for c in CATEGORICAL_FEATURES if c in demographics.columns]
        if avail_cols:
            feats = feats.merge(demographics[["household_key"] + avail_cols], on="household_key", how="left")
    feats = normalize_feature_dtypes(feats)

    drop_cols = ["last_trip_day", "last_perish_day", "primary_store_spend"]
    if "primary_store_id" not in model_features:
        drop_cols.append("primary_store_id")
    feats = feats.drop(columns=[c for c in drop_cols if c in feats.columns])

    validate_feature_frame(feats, model_features)
    return feats


class MultiDomainFeatureBuilder:
    """
    Feature (and label) factory over one loaded dataset.

    Builds trip-level baskets once and reuses them for every cutoff, so training
    (many cutoffs) and batch scoring (one cutoff) execute the identical code path.
    """

    def __init__(self, data: "RawDataBundle", lookback_days: int = 180,
                 feature_set_version: str = FEATURE_SET_VERSION) -> None:
        self.data = data
        self.lookback_days = lookback_days
        self.feature_set_version = feature_set_version
        self.features = features_for(feature_set_version)
        self._baskets: Optional[pd.DataFrame] = None

    def for_version(self, feature_set_version: str) -> "MultiDomainFeatureBuilder":
        """A builder for another feature set over the same data, sharing the basket table."""
        if feature_set_version == self.feature_set_version:
            return self
        other = MultiDomainFeatureBuilder(self.data, self.lookback_days, feature_set_version)
        other._baskets = self.baskets  # built once, shared
        return other

    @property
    def baskets(self) -> pd.DataFrame:
        if self._baskets is None:
            self._baskets = PointInTimeFeatureExtractor.build_trip_level_baskets(
                self.data.transactions, self.data.products
            )
            logger.info(f"Built {len(self._baskets):,} trip-level baskets from {self.data.source}.")
        return self._baskets

    @property
    def households(self) -> List[int]:
        return self.data.households

    def features_at(self, cutoff: int) -> pd.DataFrame:
        return compute_multidomain_features(
            baskets=self.baskets,
            cutoff=cutoff,
            households=self.households,
            campaigns=self.data.campaigns,
            redemptions=self.data.redemptions,
            demographics=self.data.demographics,
            lookback_days=self.lookback_days,
            feature_set_version=self.feature_set_version,
        )

    def labels_at(self, cutoff: int, horizon_days: int) -> pd.DataFrame:
        """
        Target = 1 if the household has >= 1 valid basket (spend > 0) in (cutoff, cutoff + horizon_days].
        """
        b = self.baskets
        future = b[(b["DAY"] > cutoff) & (b["DAY"] <= cutoff + horizon_days)]
        active = set(future["household_key"].unique())
        labels = pd.DataFrame({"household_key": self.households})
        labels["cutoff_day"] = cutoff
        labels["target"] = labels["household_key"].isin(active).astype(int)
        return labels

    def last_observed_day(self) -> int:
        return int(self.data.transactions["DAY"].max())
