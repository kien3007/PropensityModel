"""Rebuild presentation charts from the supplied Dunnhumby CSV files.

Run from the repository root: python presentations/dunnhumby_data_overview/build_figures.py
The script only writes chart files and summary.json in the presentation's figures directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "sample_data" / "dunnhumby"
OUT = Path(__file__).resolve().parent / "figures"
BLUE, TEAL, GRAY, ORANGE = "#002147", "#008b8b", "#8b97a6", "#e89b42"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                     "axes.titlesize": 18, "axes.labelsize": 16,
                     "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.labelcolor": BLUE,
                     "text.color": BLUE, "xtick.color": BLUE, "ytick.color": BLUE,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY",
            "SALES_VALUE", "STORE_ID", "RETAIL_DISC", "COUPON_DISC",
            "COUPON_MATCH_DISC"]
    t = pd.read_csv(RAW / "transaction_data.csv", usecols=cols)
    p = pd.read_csv(RAW / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT", "BRAND"])
    enriched = t.merge(p, on="PRODUCT_ID", how="left", validate="many_to_one")
    basket = t.groupby(["household_key", "BASKET_ID", "DAY"], as_index=False).agg(
        sales=("SALES_VALUE", "sum"), lines=("PRODUCT_ID", "size"))
    day_baskets = basket.groupby("DAY").size().reindex(range(1, int(t.DAY.max()) + 1), fill_value=0)
    first = t.groupby("household_key").DAY.min()
    hh_baskets = basket.groupby("household_key").size()
    hh_day = basket.groupby(["household_key", "DAY"]).size()

    fig, ax = plt.subplots(figsize=(11, 3.7))
    ax.plot(day_baskets.index, day_baskets.values, color="#cad6e1", lw=.65, label="Từng ngày")
    ax.plot(day_baskets.index, day_baskets.rolling(14, min_periods=1).mean(),
            color=BLUE, lw=2, label="Trung bình trượt 14 ngày")
    for day in (278, 643):
        if day in day_baskets.index:
            ax.annotate(f"Ngày {day}: {day_baskets.loc[day]}",
                        xy=(day, day_baskets.loc[day]), xytext=(day - 55, 85),
                        fontsize=13, color=ORANGE,
                        arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.set(xlabel="Chỉ số ngày (DAY)", ylabel="Số basket", title="Số basket theo ngày")
    ax.legend(frameon=False, loc="upper left"); save("daily_baskets")

    fig, ax = plt.subplots(figsize=(10, 3.7))
    ax.hist(first, bins=np.arange(0, 712, 14), color=BLUE, edgecolor="white")
    ax.axvline(first.median(), color=ORANGE, lw=2, label=f"Trung vị: ngày {first.median():.0f}")
    ax.set(xlabel="Ngày đầu xuất hiện trong dữ liệu", ylabel="Số hộ", title="Thời điểm hộ bắt đầu được quan sát")
    ax.legend(frameon=False); save("first_observed")

    fig, ax = plt.subplots(figsize=(10, 3.7))
    ax.hist(hh_baskets.clip(upper=600), bins=30, color=BLUE, edgecolor="white")
    ax.axvline(hh_baskets.median(), color=ORANGE, lw=2, label=f"Trung vị: {hh_baskets.median():.0f}")
    ax.axvline(hh_baskets.mean(), color=TEAL, lw=2, label=f"Trung bình: {hh_baskets.mean():.1f}")
    ax.set(xlabel="Số basket mỗi hộ (hiển thị tối đa 600)", ylabel="Số hộ",
           title="Phân bố số basket theo household")
    ax.legend(frameon=False); save("basket_per_household")

    fig, ax = plt.subplots(figsize=(10, 3.7))
    ax.hist(basket.lines.clip(upper=60), bins=np.arange(1, 63, 2), color=TEAL, edgecolor="white")
    ax.axvline(basket.lines.median(), color=ORANGE, lw=2, label=f"Trung vị: {basket.lines.median():.0f} dòng")
    ax.set(xlabel="Dòng sản phẩm / basket (hiển thị tối đa 60)", ylabel="Số basket",
           title="Một basket chứa nhiều dòng giao dịch")
    ax.legend(frameon=False); save("lines_per_basket")

    dept = enriched.groupby("DEPARTMENT", dropna=False).SALES_VALUE.sum().sort_values(ascending=False)
    dep_top = dept.head(8).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    shares = dep_top / dept.sum() * 100
    ax.barh(shares.index, shares.values, color=BLUE)
    for i, value in enumerate(shares):
        ax.text(value + .5, i, f"{value:.1f}%", va="center", color=BLUE)
    ax.set(xlabel="% tổng SALES_VALUE của tất cả department", title="8 department có doanh số ghi nhận lớn nhất")
    ax.set_xlim(0, max(shares) * 1.17); save("department_sales")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    demo = pd.read_csv(RAW / "hh_demographic.csv")
    campaign = pd.read_csv(RAW / "campaign_table.csv")
    household_set = set(t.household_key.unique())
    coverage = [len(set(demo.household_key) & household_set) / len(household_set),
                len(set(campaign.household_key) & household_set) / len(household_set)]
    bars = ax.barh(["Demographic", "Campaign assignment"], np.array(coverage)*100, color=[TEAL, BLUE])
    ax.bar_label(bars, labels=[f"{x:.2%}" for x in coverage], padding=5)
    ax.set(xlim=(0, 105), xlabel="% household có giao dịch", title="Độ phủ các bảng bổ trợ")
    save("household_coverage")

    qty_dept = enriched.loc[enriched.QUANTITY > 100].groupby("DEPARTMENT").size().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    bars = ax.barh(qty_dept.index[::-1], qty_dept.values[::-1], color=BLUE)
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set(xlabel="Số dòng QUANTITY > 100", title="Giá trị QUANTITY lớn tập trung theo department")
    ax.set_xlim(0, qty_dept.max() * 1.2); save("quantity_semantics")

    sales = enriched.groupby("BRAND").SALES_VALUE.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(sales.index, sales.values / sales.sum() * 100, color=[BLUE, TEAL])
    ax.bar_label(bars, labels=[f"{v / sales.sum():.1%}" for v in sales], padding=4)
    ax.set(ylim=(0, 90), ylabel="% SALES_VALUE", title="Doanh số theo loại nhãn hàng")
    save("brand_mix")

    summary = {"transactions": len(t), "households": t.household_key.nunique(),
               "baskets": len(basket), "household_days": len(hh_day),
               "multi_basket_household_days": int((hh_day > 1).sum()),
               "basket_zero_sales": int((basket.sales == 0).sum()),
               "basket_positive_sales": int((basket.sales > 0).sum()),
               "baskets_day_278": int(day_baskets.loc[278]),
               "baskets_day_643": int(day_baskets.loc[643]),
               "quantity_gt100_rows": int((t.QUANTITY > 100).sum()),
               "first_day_median": float(first.median()),
               "baskets_per_household_mean": float(hh_baskets.mean()),
               "baskets_per_household_median": float(hh_baskets.median()),
               "lines_per_basket_median": float(basket.lines.median()),
               "product_match_rate": float(enriched.DEPARTMENT.notna().mean())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
