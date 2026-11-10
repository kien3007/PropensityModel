"""Exploratory, train-period-only Dunnhumby analysis; does not train a model.

Outputs are written to a separate analysis directory. The user selected any
recorded basket as the seven-day target; other definitions are sensitivity
scenarios. Demographics and campaign data are described as observed; their
historical availability is not established.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = ("any_basket", "positive_basket", "grocery_department")
SNAPSHOTS = tuple(range(180, 538, 7))
HORIZON = 7


def per_household_day_matrix(
    frame: pd.DataFrame, households: np.ndarray, max_day: int, value: str
) -> np.ndarray:
    return (
        frame.pivot(index="household_key", columns="DAY", values=value)
        .reindex(index=households, columns=range(max_day + 1))
        .fillna(0)
        .to_numpy(dtype=np.int32)
    )


def make_panel(basket: pd.DataFrame, max_day: int) -> pd.DataFrame:
    """Independently construct scenario labels and historical shopping behavior."""
    daily = (
        basket.groupby(["household_key", "DAY"], observed=True)
        .agg(any_basket=("BASKET_ID", "size"),
             positive_basket=("positive_basket", "sum"),
             grocery_department=("grocery_department", "sum"),
             fuel_basket=("fuel_basket", "sum"),
             coupon_basket=("coupon_basket", "sum"),
             sales=("sales", "sum"))
        .reset_index()
    )
    households = np.sort(daily.household_key.unique())
    if max(SNAPSHOTS) + HORIZON > max_day:
        raise ValueError("The train-period target horizon is not mature")
    matrices = {
        name: per_household_day_matrix(daily, households, max_day, name)
        for name in (*TARGETS, "fuel_basket", "coupon_basket")
    }
    for matrix in matrices.values():
        if (matrix < 0).any():
            raise ValueError("Basket counts must be non-negative")
    cumulative = {name: np.cumsum(value, axis=1, dtype=np.int64)
                  for name, value in matrices.items()}
    active = matrices["any_basket"] > 0
    days = np.arange(max_day + 1, dtype=np.int16)[None, :]
    first = np.where(active, days, max_day + 1).min(axis=1)
    last = np.maximum.accumulate(np.where(active, days, -1), axis=1)
    activity = np.cumsum(active, axis=1, dtype=np.int32)
    parts = []
    for day in SNAPSHOTS:
        known = first <= day
        values = {
            "household_key": households,
            "snapshot_day": np.full(len(households), day),
            "recency_days": day - last[:, day],
            "active_days_90d": activity[:, day] - activity[:, max(day - 90, 0)],
            "fuel_baskets_90d": cumulative["fuel_basket"][:, day]
            - cumulative["fuel_basket"][:, max(day - 90, 0)],
            "coupon_baskets_90d": cumulative["coupon_basket"][:, day]
            - cumulative["coupon_basket"][:, max(day - 90, 0)],
        }
        for name in TARGETS:
            values["target_" + name] = (
                cumulative[name][:, day + HORIZON] - cumulative[name][:, day] > 0
            ).astype(np.int8)
        parts.append(pd.DataFrame(values).loc[known])
    return pd.concat(parts, ignore_index=True)


def summary_by(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_cols, observed=True, dropna=False)
        .agg(household_snapshots=("household_key", "size"),
             unique_households=("household_key", "nunique"),
             positive_household_snapshots=("target_any_basket", "sum"),
             purchase_rate=("target_any_basket", "mean"))
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("sample_data/dunnhumby"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/eda_deep_dive"))
    parser.add_argument("--existing-panel", type=Path,
                        default=Path("artifacts/propensity_v2/model/modeling_table_v2.parquet"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cols = ["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "STORE_ID",
            "SALES_VALUE", "QUANTITY", "COUPON_DISC", "COUPON_MATCH_DISC"]
    tx = pd.read_csv(args.data / "transaction_data.csv", usecols=cols)
    products = pd.read_csv(args.data / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT"])
    if products.PRODUCT_ID.duplicated().any():
        raise ValueError("Product ID is not unique")
    tx = tx.merge(products, on="PRODUCT_ID", how="left", validate="many_to_one")
    tx["is_grocery"] = tx.DEPARTMENT.eq("GROCERY")
    tx["is_fuel"] = tx.DEPARTMENT.eq("KIOSK-GAS")
    tx["coupon_line"] = (tx.COUPON_DISC < 0) | (tx.COUPON_MATCH_DISC < 0)
    basket = (
        tx.groupby(["household_key", "BASKET_ID", "DAY"], observed=True)
        .agg(sales=("SALES_VALUE", "sum"), lines=("PRODUCT_ID", "size"),
             grocery_department=("is_grocery", "max"), fuel_basket=("is_fuel", "max"),
             all_fuel=("is_fuel", "min"), coupon_basket=("coupon_line", "max"))
        .reset_index()
    )
    basket["positive_basket"] = basket.sales.gt(0)
    max_day = int(tx.DAY.max())
    panel = make_panel(basket, max_day)
    assert not panel.duplicated(["household_key", "snapshot_day"]).any()
    assert (panel.target_positive_basket <= panel.target_any_basket).all()
    assert (panel.target_grocery_department <= panel.target_any_basket).all()
    assert panel.snapshot_day.max() + HORIZON < 551

    # Compare independently built labels with a pre-existing modeling table. The
    # parquet predicate limits the comparison to the already-open training period.
    validation = {"existing_panel_checked": False}
    if args.existing_panel.exists():
        previous = pd.read_parquet(
            args.existing_panel,
            columns=["household_key", "snapshot_day", "target_purchase_next_7d"],
            filters=[("snapshot_day", "<=", max(SNAPSHOTS))],
        )
        previous = previous[previous.snapshot_day.isin(SNAPSHOTS)]
        comparison = panel.merge(previous, on=["household_key", "snapshot_day"],
                                 how="outer", validate="one_to_one", indicator=True)
        missing = int((comparison["_merge"] != "both").sum())
        disagreements = int((comparison.target_any_basket !=
                             comparison.target_purchase_next_7d).sum())
        validation = {"existing_panel_checked": True, "rows_compared": len(comparison),
                      "unmatched_keys": missing, "label_disagreements": disagreements}
        if missing or disagreements:
            raise AssertionError(f"Independent label validation failed: {validation}")

    # The sample is checked directly against raw basket records, not the dense
    # cumulative arrays used to construct the panel.
    for row in panel.sample(min(100, len(panel)), random_state=42).itertuples(index=False):
        future = basket[(basket.household_key == row.household_key) &
                        (basket.DAY > row.snapshot_day) &
                        (basket.DAY <= row.snapshot_day + HORIZON)]
        assert int(not future.empty) == row.target_any_basket
        assert int(future.positive_basket.any()) == row.target_positive_basket
        assert int(future.grocery_department.any()) == row.target_grocery_department
    validation["raw_basket_sample_checks"] = min(100, len(panel))

    scenarios = []
    for name in TARGETS:
        scenarios.append({"definition": name, "household_snapshots": len(panel),
                          "positive_count": int(panel["target_" + name].sum()),
                          "positive_rate": float(panel["target_" + name].mean()),
                          "diff_from_any": int((panel.target_any_basket !=
                                                panel["target_" + name]).sum())})
    pd.DataFrame(scenarios).to_csv(args.output / "target_scenarios.csv", index=False)
    weekly = panel.groupby("snapshot_day", observed=True).agg(
        households=("household_key", "size"),
        any_basket=("target_any_basket", "mean"),
        positive_basket=("target_positive_basket", "mean"),
        grocery_department=("target_grocery_department", "mean"),
    ).reset_index()
    weekly.to_csv(args.output / "target_scenarios_by_snapshot.csv", index=False)

    panel["recency_group"] = pd.cut(
        panel.recency_days, [-1, 7, 14, 30, 60, 90, np.inf],
        labels=["0-7", "8-14", "15-30", "31-60", "61-90", "91+"])
    panel["activity_group"] = pd.cut(
        panel.active_days_90d, [-1, 0, 3, 9, 18, np.inf],
        labels=["0", "1-3", "4-9", "10-18", "19+"])
    panel["fuel_90d"] = np.where(panel.fuel_baskets_90d > 0, "có fuel", "không fuel")
    panel["coupon_90d"] = np.where(panel.coupon_baskets_90d > 0,
                                   "có coupon discount", "không coupon discount")
    for name in ("recency_group", "activity_group", "fuel_90d", "coupon_90d"):
        summary_by(panel, [name]).to_csv(args.output / f"segment_{name}.csv", index=False)
    for name in ("fuel_90d", "coupon_90d"):
        summary_by(panel, ["activity_group", name]).to_csv(
            args.output / f"stratified_activity_{name}.csv", index=False)

    demographics = pd.read_csv(args.data / "hh_demographic.csv", usecols=["household_key"])
    panel["demographics_available"] = np.where(
        panel.household_key.isin(demographics.household_key), "có hồ sơ", "không có hồ sơ")
    summary_by(panel, ["demographics_available"]).to_csv(
        args.output / "demographic_coverage_descriptive.csv", index=False)
    summary_by(panel, ["activity_group", "demographics_available"]).to_csv(
        args.output / "demographic_coverage_by_activity.csv", index=False)

    # Redemption events carry their own DAY and are restricted to the 90 days
    # preceding the cutoff; this is still an association, not an exposure test.
    redemption = pd.read_csv(args.data / "coupon_redempt.csv", usecols=["household_key", "DAY"])
    redemption_days = {hh: np.sort(g.DAY.to_numpy())
                       for hh, g in redemption.groupby("household_key")}
    redemption_flags = np.zeros(len(panel), dtype=bool)
    for household, index in panel.groupby("household_key").groups.items():
        observed_days = redemption_days.get(household)
        if observed_days is None:
            continue
        cutoffs = panel.loc[index, "snapshot_day"].to_numpy()
        right = np.searchsorted(observed_days, cutoffs, side="right")
        left = np.searchsorted(observed_days, cutoffs - 90, side="right")
        redemption_flags[np.asarray(index)] = right > left
    panel["redemption_90d"] = np.where(redemption_flags, "có redemption", "không redemption")
    summary_by(panel, ["redemption_90d"]).to_csv(
        args.output / "segment_redemption_90d.csv", index=False)
    summary_by(panel, ["activity_group", "redemption_90d"]).to_csv(
        args.output / "stratified_activity_redemption_90d.csv", index=False)

    hh_first = tx.groupby("household_key").DAY.min()
    days = basket.groupby("DAY").agg(baskets=("BASKET_ID", "size"),
                                      households=("household_key", "nunique"),
                                      sales=("sales", "sum"))
    anomalies = days.loc[[278, 643]].reset_index()
    anomalies.to_csv(args.output / "low_volume_days.csv", index=False)
    qty = (tx[tx.QUANTITY.gt(100)].groupby("DEPARTMENT", dropna=False)
           .agg(rows=("QUANTITY", "size"), median_quantity=("QUANTITY", "median"),
                max_quantity=("QUANTITY", "max"), sales=("SALES_VALUE", "sum"))
           .sort_values("rows", ascending=False).reset_index())
    qty.to_csv(args.output / "quantity_gt100_by_department.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, label in [("any_basket", "Mọi basket"),
                       ("positive_basket", "Basket sales > 0"),
                       ("grocery_department", "Có department GROCERY")]:
        ax.plot(weekly.snapshot_day, weekly[col], label=label)
    ax.set(xlabel="Snapshot day (chỉ giai đoạn train)", ylabel="Tỷ lệ dương",
           title="Độ nhạy nhãn theo định nghĩa basket")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(args.output / "target_scenarios_by_snapshot.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [(axes[0], "recency_group", "Recency"),
                            (axes[1], "activity_group", "Ngày hoạt động 90 ngày")]:
        group = summary_by(panel, [col])
        ax.bar(group[col].astype(str), group.purchase_rate, color="#17618a")
        ax.set(title=title, ylabel="Tỷ lệ có basket trong 7 ngày", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(args.output / "segments.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days.index, days.baskets, color="#6f94aa", linewidth=.7)
    for day in (278, 643):
        ax.scatter(day, days.loc[day, "baskets"], color="#d47b24", zorder=3)
        ax.annotate(f"{day}: {days.loc[day, 'baskets']}",
                    (day, days.loc[day, "baskets"]), xytext=(day - 45, 110),
                    arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="DAY", ylabel="Basket/ngày", title="Số basket theo ngày — toàn bộ dữ liệu")
    fig.tight_layout(); fig.savefig(args.output / "daily_anomalies.png", dpi=160); plt.close(fig)

    metadata = {
        "purpose": "Exploratory data understanding, not model training",
        "snapshot_scope": "train snapshots days 180–537 inclusive; label days through 544",
        "horizon_days": HORIZON,
        "selected_target": "any_basket",
        "target_decision": (
            "User confirmed any recorded BASKET_ID in (snapshot, snapshot+7], "
            "including zero-sales, fuel and miscellaneous baskets."
        ),
        "quantity_decision": (
            "Exclude QUANTITY from cross-department behavior aggregation while "
            "units remain unverified; retain it in raw-data quality analysis."
        ),
        "rows": len(panel), "unique_households": int(panel.household_key.nunique()),
        "transactions": len(tx), "baskets": len(basket),
        "first_observed_day_median": float(hh_first.median()),
        "baskets_zero_sales": int((basket.sales == 0).sum()),
        "baskets_only_fuel": int(basket.all_fuel.sum()),
        "validation": validation,
        "scenarios": scenarios,
        "definitions": {
            "any_basket": "at least one recorded BASKET_ID in (snapshot, snapshot+7]",
            "positive_basket": "at least one basket whose sum of SALES_VALUE > 0 in the horizon",
            "grocery_department": "at least one basket containing department GROCERY; excludes other food departments",
        },
        "cautions": [
            "Outcomes describe purchases recorded by this retailer, not shopping elsewhere.",
            "Household-snapshot observations repeat; counts and rates are not independent people.",
            "Demographics are described retrospectively; availability at cutoff is unverified.",
            "Coupon discount is a transaction signal, not proof of causal effect or exposure.",
            "High quantity values are observed; units or scaling for fuel remain unverified.",
            "SALES_VALUE is retailer receipts and may differ from customer out-of-pocket payment.",
            "Days 278 and 643 have one basket each; source or calendar cause is unverified.",
        ],
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
