
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Legacy 23 features
MODEL_FEATURES: List[str] = [
    "recency_days",
    "observed_tenure_days",
    "baskets_7d",
    "sales_7d",
    "line_items_7d",
    "active_days_7d",
    "baskets_30d",
    "sales_30d",
    "line_items_30d",
    "active_days_30d",
    "baskets_90d",
    "sales_90d",
    "line_items_90d",
    "active_days_90d",
    "baskets_180d",
    "sales_180d",
    "line_items_180d",
    "active_days_180d",
    "cadence_median_gap_180d",
    "cadence_gap_std_180d",
    "cadence_recency_ratio",
    "cadence_has_gap",
    "cadence_has_two_gaps",
]

# Champion 28 Multi-Domain Features (from projectphase3)
MULTIDOMAIN_FEATURES_28: List[str] = [
    "days_since_last_basket",
    "days_since_last_perishable",
    "median_inter_trip",
    "std_inter_trip",
    "urgency_ratio",
    "baskets_count_14d",
    "baskets_count_30d",
    "baskets_count_60d",
    "baskets_count_90d",
    "baskets_count_180d",
    "frequency_momentum_30_vs_90",
    "spend_30d",
    "spend_90d",
    "spend_180d",
    "aov_90d",
    "avg_items_per_basket_90d",
    "perishable_basket_ratio_90d",
    "is_targeted_by_active_campaign",
    "coupon_redemption_count_180d",
    "primary_store_id",
    "store_loyalty_index",
    "classification_1",
    "classification_2",
    "classification_3",
    "HOMEOWNER_DESC",
    "classification_4",
    "classification_5",
    "KID_CATEGORY_DESC",
]

CATEGORICAL_FEATURES_28: List[str] = [
    "classification_1",
    "classification_2",
    "classification_3",
    "HOMEOWNER_DESC",
    "classification_4",
    "classification_5",
    "KID_CATEGORY_DESC",
]

PERISHABLE_DEPARTMENTS: List[str] = [
    "PRODUCE", "MEAT", "MEAT-PCKGD", "DELI", "PASTRY", "DAIRY"
]


class PointInTimeFeatureExtractor:
    """
    Extracts Point-in-Time behavioral, RFM, and cadence features as of cutoff date t_0.
    """

    @staticmethod
    def extract_features(
        transactions: pd.DataFrame,
        cutoff: int,
        feature_cols: List[str] = MODEL_FEATURES,
    ) -> pd.DataFrame:
        """
        Extracts features strictly for transactions where DAY <= cutoff.
        """
        source = transactions.loc[transactions["DAY"] <= cutoff].copy()
        if source.empty:
            raise ValueError(f"No transactions found on or before cutoff DAY = {cutoff}")

        # Validate basket uniqueness per household and day
        scope = source.groupby("BASKET_ID").agg(
            households=("household_key", "nunique"), days=("DAY", "nunique")
        )
        if not (scope["households"].eq(1).all() and scope["days"].eq(1).all()):
            raise ValueError("Basket ID is not unique per household and day")

        basket = (
            source.groupby(["household_key", "BASKET_ID", "DAY"], as_index=False, sort=False)
            .agg(sales=("SALES_VALUE", "sum"), line_items=("SALES_VALUE", "size"))
        )

        daily = (
            basket.groupby(["household_key", "DAY"], as_index=False)
            .agg(baskets=("BASKET_ID", "size"), sales=("sales", "sum"), line_items=("line_items", "sum"))
        )

        first = daily.groupby("household_key")["DAY"].min()
        last = daily.groupby("household_key")["DAY"].max()

        snapshot = pd.DataFrame(
            {
                "household_key": first.index,
                "snapshot_day": cutoff,
                "recency_days": cutoff - last.to_numpy(),
                "observed_tenure_days": cutoff - first.to_numpy(),
            }
        ).set_index("household_key")

        # Rolling window metrics
        for w in (7, 30, 90, 180):
            window = daily.loc[(daily["DAY"] > cutoff - w) & (daily["DAY"] <= cutoff)]
            agg = window.groupby("household_key").agg(
                baskets=("baskets", "sum"),
                sales=("sales", "sum"),
                line_items=("line_items", "sum"),
                active_days=("DAY", "size"),
            )
            for col in ["baskets", "sales", "line_items", "active_days"]:
                snapshot[f"{col}_{w}d"] = agg[col].reindex(snapshot.index).fillna(0)

        # Precision float64 cumulative sales matching data prep contract
        sales_matrix = (
            daily.pivot(index="household_key", columns="DAY", values="sales")
            .reindex(index=snapshot.index, columns=range(cutoff + 1))
            .fillna(0)
            .to_numpy(dtype=np.float64)
        )
        cumulative_sales = np.cumsum(sales_matrix, axis=1)
        for w in (7, 30, 90, 180):
            snapshot[f"sales_{w}d"] = cumulative_sales[:, cutoff] - cumulative_sales[:, max(cutoff - w, 0)]

        # Cadence features
        days = daily.loc[
            (daily["DAY"] > cutoff - 180) & (daily["DAY"] <= cutoff), ["household_key", "DAY"]
        ].sort_values(["household_key", "DAY"]).copy()
        days["gap"] = days.groupby("household_key")["DAY"].diff()
        gap = days.groupby("household_key")["gap"].agg(["count", "median", "std"])

        snapshot["cadence_median_gap_180d"] = gap["median"]
        snapshot["cadence_gap_std_180d"] = gap["std"]
        snapshot["cadence_has_gap"] = gap["count"].ge(1).reindex(snapshot.index, fill_value=False).astype(int)
        snapshot["cadence_has_two_gaps"] = gap["count"].ge(2).reindex(snapshot.index, fill_value=False).astype(int)
        snapshot["cadence_recency_ratio"] = snapshot["recency_days"] / snapshot["cadence_median_gap_180d"]

        snapshot = snapshot.reset_index()

        # Guarantee full universe of 2,500 households
        all_hh = pd.DataFrame({"household_key": list(range(1, 2501))})
        snapshot = all_hh.merge(snapshot, on="household_key", how="left")
        snapshot["snapshot_day"] = snapshot["snapshot_day"].fillna(cutoff).astype(int)
        snapshot["recency_days"] = snapshot["recency_days"].fillna(cutoff).astype(float)
        snapshot["observed_tenure_days"] = snapshot["observed_tenure_days"].fillna(cutoff).astype(float)

        for col in feature_cols:
            if col not in snapshot.columns:
                snapshot[col] = 0.0
            snapshot[col] = snapshot[col].fillna(0.0)

        assert not snapshot.duplicated(["household_key", "snapshot_day"]).any()
        assert set(feature_cols).issubset(snapshot.columns)
        assert not snapshot[feature_cols].isin([np.inf, -np.inf]).any().any()

        return snapshot

    @staticmethod
    def build_trip_level_baskets(
        transactions: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        perishable_departments: List[str] = PERISHABLE_DEPARTMENTS,
    ) -> pd.DataFrame:
        """
        Aggregates raw item transactions into trip-level Baskets with perishable indicators.
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

    @staticmethod
    def extract_multidomain_features(
        cutoff: int,
        raw_data_dir: Optional[Path | str] = None,
        baskets: Optional[pd.DataFrame] = None,
        campaigns: Optional[pd.DataFrame] = None,
        redemptions: Optional[pd.DataFrame] = None,
        demographics: Optional[pd.DataFrame] = None,
        lookback_days: int = 180,
    ) -> pd.DataFrame:
        """
        Extracts 28 multi-domain features (Perishable, Urgency, Momentum, Marketing, Store Loyalty, Demographics).
        Strictly Point-in-Time (DAY <= cutoff).
        """
        raw_dir = Path(raw_data_dir or settings.DATA_RAW_DIR)

        if baskets is None:
            tx_path = raw_dir / "transaction_data.csv"
            prod_path = raw_dir / "product.csv"
            tx = pd.read_csv(tx_path)
            prod = pd.read_csv(prod_path) if prod_path.exists() else None
            baskets = PointInTimeFeatureExtractor.build_trip_level_baskets(tx, prod)

        if campaigns is None:
            camp_desc_path = raw_dir / "campaign_desc.csv"
            camp_table_path = raw_dir / "campaign_table.csv"
            if camp_desc_path.exists() and camp_table_path.exists():
                cd = pd.read_csv(camp_desc_path)
                ct = pd.read_csv(camp_table_path)
                campaigns = ct.merge(cd, on="CAMPAIGN", how="inner")

        if redemptions is None:
            red_path = raw_dir / "coupon_redempt.csv"
            if red_path.exists():
                redemptions = pd.read_csv(red_path)

        if demographics is None:
            demo_path = raw_dir / "hh_demographic.csv"
            if demo_path.exists():
                demographics = pd.read_csv(demo_path)

        hist = baskets[(baskets["DAY"] <= cutoff) & (baskets["DAY"] >= cutoff - lookback_days)].copy()
        all_households = pd.DataFrame({"household_key": list(range(1, 2501))})
        feats = all_households.copy()
        feats["cutoff_day"] = cutoff

        # A. Recency & Timing
        latest_trip = hist.groupby("household_key")["DAY"].max().reset_index()
        latest_trip.rename(columns={"DAY": "last_trip_day"}, inplace=True)
        feats = feats.merge(latest_trip, on="household_key", how="left")
        feats["days_since_last_basket"] = (cutoff - feats["last_trip_day"]).fillna(lookback_days + 1)

        perish_hist = hist[hist["has_perishable"] == True]
        if not perish_hist.empty:
            latest_perish = perish_hist.groupby("household_key")["DAY"].max().reset_index()
            latest_perish.rename(columns={"DAY": "last_perish_day"}, inplace=True)
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
            sub = hist[hist["DAY"] >= cutoff - window]
            counts = sub.groupby("household_key")["BASKET_ID"].nunique().reset_index()
            counts.columns = ["household_key", f"baskets_count_{window}d"]
            feats = feats.merge(counts, on="household_key", how="left")
            feats[f"baskets_count_{window}d"] = feats[f"baskets_count_{window}d"].fillna(0)

        feats["frequency_momentum_30_vs_90"] = (
            (feats["baskets_count_30d"] / 30.0) /
            ((feats["baskets_count_90d"] / 90.0) + 1e-5)
        )

        # C. Monetary & Basket Architecture
        for window in [30, 90, 180]:
            sub = hist[hist["DAY"] >= cutoff - window]
            spend = sub.groupby("household_key")["basket_spend"].sum().reset_index()
            spend.columns = ["household_key", f"spend_{window}d"]
            feats = feats.merge(spend, on="household_key", how="left")
            feats[f"spend_{window}d"] = feats[f"spend_{window}d"].fillna(0)

        feats["aov_90d"] = feats["spend_90d"] / (feats["baskets_count_90d"] + 1e-5)

        sub90 = hist[hist["DAY"] >= cutoff - 90]
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
            hist_red = redemptions[(redemptions["DAY"] <= cutoff) & (redemptions["DAY"] >= cutoff - lookback_days)]
            red_counts = hist_red.groupby("household_key")["DAY"].count().reset_index()
            red_counts.columns = ["household_key", "coupon_redemption_count_180d"]
            feats = feats.merge(red_counts, on="household_key", how="left")
            feats["coupon_redemption_count_180d"] = feats["coupon_redemption_count_180d"].fillna(0)
        else:
            feats["coupon_redemption_count_180d"] = 0

        # E. Store Loyalty
        if not hist.empty and "STORE_ID" in hist.columns:
            store_agg = hist.groupby(["household_key", "STORE_ID"])["basket_spend"].sum().reset_index()
            primary_store = store_agg.sort_values(["household_key", "basket_spend"], ascending=[True, False]).groupby("household_key").first().reset_index()
            primary_store.rename(columns={"STORE_ID": "primary_store_id", "basket_spend": "primary_store_spend"}, inplace=True)
            feats = feats.merge(primary_store[["household_key", "primary_store_id", "primary_store_spend"]], on="household_key", how="left")
            feats["store_loyalty_index"] = (feats["primary_store_spend"] / (feats["spend_180d"] + 1e-5)).fillna(0).clip(0, 1)
            feats["primary_store_id"] = feats["primary_store_id"].fillna(0).astype(int)
        else:
            feats["primary_store_id"] = 0
            feats["store_loyalty_index"] = 0

        # F. Demographics
        if demographics is not None and not demographics.empty:
            avail_cols = [c for c in CATEGORICAL_FEATURES_28 if c in demographics.columns]
            if avail_cols:
                feats = feats.merge(demographics[["household_key"] + avail_cols], on="household_key", how="left")

        for c in CATEGORICAL_FEATURES_28:
            if c not in feats.columns:
                feats[c] = "UNKNOWN"
            feats[c] = feats[c].fillna("UNKNOWN").astype("category")

        drop_cols = ["last_trip_day", "last_perish_day", "primary_store_spend"]
        feats.drop(columns=[c for c in drop_cols if c in feats.columns], inplace=True)

        return feats
