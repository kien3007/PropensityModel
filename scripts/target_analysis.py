"""
Target & Business-Value Analysis (Phase 3).

1. How predictable each target is: purchase base rate by horizon and by frequency segment.
2. For every target (purchase, lapse): out-of-time LightGBM metrics and the targeting lists it
   produces at fixed weekly campaign capacities (top 5 / 10 / 20% of scored households per cutoff).
   - purchase value = spend the selected households make inside the horizon
   - lapse value    = 90-day spend (pre-cutoff, i.e. known at decision time) of correctly
                      flagged lapsers: the revenue run-rate a retention action could protect
3. Writes docs/analysis/TARGET_ANALYSIS.md and the raw capacity table as CSV.

Usage: python -m scripts.target_analysis [--data-source lakehouse|csv]
Nothing is promoted: models are trained with promote=False.
"""
from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config.model_config import TRAINING_CONFIGS
from src.config.settings import settings
from src.data.loader import DataLoader, RawDataBundle
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.pipelines.train_pipeline import run_training_pipeline

logger = logging.getLogger("TargetAnalysis")

CAPACITIES = (0.05, 0.10, 0.20)
DOCS_DIR = settings.BASE_DIR / "docs" / "analysis"


def predictability_tables(builder: MultiDomainFeatureBuilder) -> Dict[str, pd.DataFrame]:
    cutoffs = [200, 260, 320, 380, 440, 500, 560, 620, 680]
    last_day = builder.last_observed_day()
    horizon_rows = []
    for h in (3, 7, 14, 21, 28):
        rates = [builder.labels_at(c, h)["target"].mean() for c in cutoffs if c + h <= last_day]
        horizon_rows.append({"horizon_days": str(h), "mean_rate": np.mean(rates), "min_rate": min(rates),
                             "max_rate": max(rates), "max_lift_10": 1 / np.mean(rates)})

    frames = []
    for c in (500, 560, 620, 680):
        frames.append(builder.features_at(c).merge(builder.labels_at(c, 14), on=["household_key", "cutoff_day"]))
    stacked = pd.concat(frames)
    segment = pd.cut(stacked["baskets_count_90d"], [-1, 0, 3, 9, 19, np.inf], labels=["0", "1-3", "4-9", "10-19", "20+"])
    seg_table = (stacked.groupby(segment, observed=True)["target"].agg(["mean", "size"])
                 .rename(columns={"mean": "purchase_rate_14d", "size": "rows"}).reset_index())
    seg_table["share_of_rows"] = seg_table["rows"] / seg_table["rows"].sum()
    return {"horizon": pd.DataFrame(horizon_rows), "segment": seg_table}


def future_spend(builder: MultiDomainFeatureBuilder, cutoff: int, horizon: int) -> pd.Series:
    b = builder.baskets
    window = b[(b["DAY"] > cutoff) & (b["DAY"] <= cutoff + horizon)]
    return window.groupby("household_key")["basket_spend"].sum()


def capacity_table(preds: pd.DataFrame, builder: MultiDomainFeatureBuilder, target: str, horizon: int) -> pd.DataFrame:
    """
    LightGBM targeting lists at fixed capacities, ranked the way production ranks them:
    purchase by p(purchase); lapse by expected spend at risk = p(lapse) x 90-day spend
    (the batch retention list).
    """
    rows: List[dict] = []
    for cutoff, grp in preds.groupby("cutoff_day"):
        grp = grp.copy()
        if target == "purchase":
            grp["value"] = grp["household_key"].map(future_spend(builder, cutoff, horizon)).fillna(0.0)
            grp["rank_score"] = grp["model_probability"]
        else:
            spend90 = grp["household_key"].map(
                builder.features_at(cutoff).set_index("household_key")["spend_90d"]).fillna(0.0)
            grp["value"] = spend90 * grp["target"]
            grp["rank_score"] = grp["model_probability"] * spend90
        positives, total_value = grp["target"].sum(), grp["value"].sum()
        for cap in CAPACITIES:
            k = max(1, int(round(cap * len(grp))))
            top = grp.nlargest(k, "rank_score")
            rows.append({"capacity": cap, "k": k, "hits": int(top["target"].sum()), "positives": int(positives),
                         "value": float(top["value"].sum()), "total_value": float(total_value)})
    agg = pd.DataFrame(rows).groupby("capacity").sum(numeric_only=True).reset_index()
    agg["precision"] = agg["hits"] / agg["k"]
    agg["capture_rate"] = agg["hits"] / agg["positives"]
    agg["value_share"] = agg["value"] / agg["total_value"]
    agg.insert(0, "target", target)
    return agg


def md_table(df: pd.DataFrame, floatfmt: Dict[str, str]) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt[c]) if c in floatfmt and isinstance(v, (int, float, np.number)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


RANKING = {"purchase": "p(purchase)", "lapse": "p(lapse) x 90-day spend (production retention ranking)"}
VALUE_LABEL = {"purchase": "spend in horizon ($)", "lapse": "90d spend of caught lapsers ($)"}


def build_report(data: RawDataBundle, out_dir: Path) -> Path:
    builder = MultiDomainFeatureBuilder(data)
    pred_tables = predictability_tables(builder)
    sections, capacity_frames = [], []
    for target, cfg in TRAINING_CONFIGS.items():
        summary = run_training_pipeline(config=cfg, data=data, output_dir=out_dir / target, promote=False)
        preds = pd.read_parquet(summary["test_predictions_path"])
        cap = capacity_table(preds, builder, target, cfg.horizon_days)
        capacity_frames.append(cap)

        m = summary["test_metrics"]
        metrics = pd.DataFrame([{k: m[k] for k in ("roc_auc", "pr_auc", "lift_at_10pct", "brier_skill_score", "ece")}])
        view = cap[["capacity", "k", "precision", "capture_rate", "value", "value_share"]].copy()
        view["capacity"] = (view["capacity"] * 100).map(lambda x: f"top {x:.0f}%")
        view = view.rename(columns={"value": VALUE_LABEL[target], "k": "households"})
        sections.append(
            f"### Target `{target}` (horizon {cfg.horizon_days} days, test cutoffs {list(cfg.test_cutoffs)})\n\n"
            f"Test rows: {summary['rows']['test']:,}; base rate {m['base_rate']:.3f}; "
            f"max attainable lift@10% {m['max_attainable_lift_10']:.2f}x.\n\n"
            + md_table(metrics, {"roc_auc": ".4f", "pr_auc": ".4f", "lift_at_10pct": ".2f",
                                 "brier_skill_score": ".3f", "ece": ".4f"})
            + f"\n\nTargeting lists ranked by {RANKING[target]}, summed over test cutoffs:\n\n"
            + md_table(view, {"precision": ".3f", "capture_rate": ".3f", VALUE_LABEL[target]: ",.0f",
                              "value_share": ".3f"})
        )

    capacity = pd.concat(capacity_frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    capacity.to_csv(out_dir / "capacity_scenarios.csv", index=False)

    horizon = pred_tables["horizon"]
    report = f"""# Target & Business-Value Analysis

> Generated by `python -m scripts.target_analysis` on {datetime.date.today().isoformat()} from {data.source}
> (data fingerprint `{data.fingerprint()}`). All numbers are out-of-time test results of the LightGBM models;
> nothing was promoted.

## 1. How predictable is "will the household buy?"

Purchase base rate by horizon (mean over 9 cutoffs between DAY 200 and 680):

{md_table(horizon, {"mean_rate": ".3f", "min_rate": ".3f", "max_rate": ".3f", "max_lift_10": ".2f"})}

14-day purchase rate by shopping frequency in the previous 90 days (cutoffs 500-680):

{md_table(pred_tables["segment"], {"purchase_rate_14d": ".3f", "share_of_rows": ".3f"})}

At 14 days most households buy, and frequency alone almost decides the label. The ceiling on
top-decile lift is ~1.4x: LightGBM ranks these households well (high AUC), but the top decile is
mostly people who were going to shop anyway.

## 2. LightGBM performance and targeting lists

{chr(10).join(chr(10) + s + chr(10) for s in sections)}
## 3. Reading the results

- **purchase**: the top of the list is almost entirely regular shoppers (precision close to 1.0).
  Their in-horizon spend is not incremental without a holdout, so it is not campaign value.
- **lapse**: the base rate is ~11%, so ranking matters. The retention list is ranked by expected
  spend at risk, p(lapse) x 90-day spend, which is the ranking the batch pipeline exports.
- Value columns measure *association*, not causal uplift. A holdout / uplift design is needed to
  measure campaign impact.

Raw numbers: `data/outputs/analysis/capacity_scenarios.csv`.
"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "TARGET_ANALYSIS.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-source", choices=["lakehouse", "csv"], default=None)
    parser.add_argument("--output-dir", type=Path, default=settings.OUTPUT_DIR / "analysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    path = build_report(DataLoader.load(args.data_source), args.output_dir)
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
