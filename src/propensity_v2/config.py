"""Configuration and feature contracts for propensity V2."""

from __future__ import annotations

from dataclasses import dataclass


DEMOGRAPHIC_COLUMNS = (
    "classification_1",
    "classification_2",
    "classification_3",
    "HOMEOWNER_DESC",
    "classification_5",
    "classification_4",
    "KID_CATEGORY_DESC",
)

WINDOWS = (7, 30, 90, 180)

ADDITIVE_DAILY_METRICS = (
    "spend",
    "baskets",
    "line_items",
    "active_days",
    "discount",
    "coupon_discount",
    "discounted_lines",
    "private_label_spend",
    "grocery_spend",
    "fresh_spend",
    "drug_gm_spend",
    "fuel_spend",
    "other_spend",
    "morning_baskets",
    "afternoon_baskets",
    "evening_baskets",
    "product_day_diversity",
    "department_day_diversity",
    "store_day_diversity",
)


@dataclass(frozen=True)
class PropensityV2Config:
    horizon_days: int = 7
    snapshot_step_days: int = 7
    minimum_history_days: int = 180
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    top_k_fraction: float = 0.10
    random_state: int = 42
    bootstrap_repeats: int = 300

    @property
    def target_column(self) -> str:
        return f"target_purchase_next_{self.horizon_days}d"
